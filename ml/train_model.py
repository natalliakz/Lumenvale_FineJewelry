"""Fit the Lumenvale marketing mix model once and publish its outputs.

Steps
1. Read the weekly inputs from LUMENVALE_MMM.PUBLIC (or the local synthetic
   CSVs when Snowflake is not available; see mmm_data.py).
2. Back-test: fit on all but the last 26 weeks and score the held-out weeks.
3. Fit on all weeks.
4. Write response curves, contributions, ROI and marginal ROI, fit diagnostics,
   posterior draws and the run metadata (MMM_MODEL_RUN) to Snowflake, with local
   copies in outputs/.

The Streamlit app only reads these outputs, so it never refits anything.

Run:  uv run python ml/train_model.py                            # Snowflake if available
      MMM_DATA_SOURCE=snowflake uv run python ml/train_model.py   # Snowflake or fail
      MMM_DATA_SOURCE=local uv run python ml/train_model.py       # files only

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "ml"))

from mmm_data import TRUTH_TABLE, TableReader, load_model_inputs  # noqa: E402
from model_utils import (  # noqa: E402
    CHANNELS, build_model, design, draws_in_dollars, marginal, predicted_revenue, saturation,
)

MODEL_VERSION = "v3.2"
HOLDOUT_WEEKS = 26
RUN_RATE_WEEKS = 52  # "current plan" = trailing 52-week average weekly spend
QUARTER_WEEKS = 13
SAMPLER = dict(draws=1000, tune=1500, chains=4, target_accept=0.95, random_seed=42,
               progressbar=False)


def q(a, p, axis=0):
    return np.quantile(a, p, axis=axis)


def fit(d: dict, n_train: int | None = None):
    with build_model(d, n_train):
        return pm.sample(**SAMPLER)


def main() -> None:
    t0 = time.time()
    reader = TableReader()
    print(f"Reading inputs from: {reader.label}")
    df = load_model_inputs(reader)
    d = design(df)
    n = len(df)

    # --- 1. Back-test on the last 26 weeks ------------------------------------------
    print(f"Back-test fit on {n - HOLDOUT_WEEKS} weeks ...")
    idata_bt = fit(d, n - HOLDOUT_WEEKS)
    pred_bt, _ = predicted_revenue(draws_in_dollars(idata_bt, d), d)
    hold = slice(n - HOLDOUT_WEEKS, n)
    actual = df["revenue"].to_numpy()
    holdout_mape = float(np.mean(np.abs(pred_bt[:, hold].mean(0) / actual[hold] - 1)))
    print(f"  hold-out MAPE: {holdout_mape:.1%}")

    # --- 2. Full fit -----------------------------------------------------------------
    print(f"Full fit on {n} weeks ...")
    idata = fit(d)
    rhat = float(max(np.nanmax(v.values) for v in pm.stats.rhat(idata).data_vars.values()))
    draws = draws_in_dollars(idata, d)
    pred, media = predicted_revenue(draws, d)  # (draw, week), (draw, week, channel)
    pred_mean = pred.mean(0)
    r2 = float(1 - np.sum((actual - pred_mean) ** 2) / np.sum((actual - actual.mean()) ** 2))
    in_mape = float(np.mean(np.abs(pred_mean / actual - 1)))
    print(f"  in-sample R2 {r2:.3f}, MAPE {in_mape:.1%}, max R-hat {rhat:.3f}")

    weeks = df["week_start"]
    outputs: dict[str, pd.DataFrame] = {}

    # Posterior draws: what the app uses to project any scenario with uncertainty.
    outputs["MMM_POSTERIOR_DRAWS"] = pd.DataFrame([
        dict(draw=i, channel=ch, decay=draws["decay"][i, c],
             k=draws["k_dollars"][i, c], beta=draws["beta_dollars"][i, c])
        for i in range(len(draws["decay"])) for c, ch in enumerate(CHANNELS)
    ])

    # Fit: actual vs predicted.
    outputs["MMM_FIT"] = pd.DataFrame({
        "week_start": weeks, "actual": actual, "predicted": pred_mean,
        "predicted_p05": q(pred, 0.05), "predicted_p95": q(pred, 0.95),
        "in_holdout_backtest": np.arange(n) >= n - HOLDOUT_WEEKS,
        "backtest_predicted": pred_bt.mean(0),
    })

    # Weekly contributions: each channel plus the non-media baseline.
    contrib = pd.DataFrame(media.mean(0), columns=CHANNELS)
    contrib["Baseline"] = pred_mean - media.mean(0).sum(1)
    contrib["week_start"] = weeks
    outputs["MMM_CONTRIBUTIONS_WEEKLY"] = contrib.melt(
        id_vars="week_start", var_name="component", value_name="contribution")

    # Channel summary: ROI over the whole history, marginal ROI at today's run rate.
    spend = d["spend"]
    run_rate = spend[-RUN_RATE_WEEKS:].mean(0)
    k, beta = draws["k_dollars"], draws["beta_dollars"]
    total_contrib = media.sum(1)  # (draw, channel)
    roi = total_contrib / spend.sum(0)
    mroi = marginal(run_rate[None, :], k, beta)
    rows = []
    for c, ch in enumerate(CHANNELS):
        rows.append(dict(
            channel=ch,
            total_spend=spend[:, c].sum(),
            contribution=total_contrib[:, c].mean(),
            contribution_p05=q(total_contrib[:, c], 0.05),
            contribution_p95=q(total_contrib[:, c], 0.95),
            roi=roi[:, c].mean(), roi_p05=q(roi[:, c], 0.05), roi_p95=q(roi[:, c], 0.95),
            run_rate_weekly_spend=run_rate[c],
            current_quarter_budget=run_rate[c] * QUARTER_WEEKS,
            max_observed_weekly_spend=spend[:, c].max(),
            marginal_roi=mroi[:, c].mean(),
            marginal_roi_p05=q(mroi[:, c], 0.05), marginal_roi_p95=q(mroi[:, c], 0.95),
            decay=draws["decay"][:, c].mean(),
        ))
    outputs["MMM_CHANNEL_SUMMARY"] = pd.DataFrame(rows)

    # Response curves: weekly revenue vs weekly spend, with a 90% band.
    curves = []
    for c, ch in enumerate(CHANNELS):
        top = max(3 * run_rate[c], 1.2 * spend[:, c].max())
        grid = np.linspace(0, top, 80)
        rev = saturation(grid[None, :], k[:, c, None], beta[:, c, None])
        mr = marginal(grid[None, :], k[:, c, None], beta[:, c, None])
        curves.append(pd.DataFrame({
            "channel": ch, "weekly_spend": grid,
            "revenue": rev.mean(0), "revenue_p05": q(rev, 0.05), "revenue_p95": q(rev, 0.95),
            "marginal_roi": mr.mean(0),
            "within_observed_range": grid <= spend[:, c].max(),
        }))
    outputs["MMM_RESPONSE_CURVES"] = pd.concat(curves, ignore_index=True)

    # Recovered vs true parameters (only possible because the data is synthetic).
    truth = reader.read(TRUTH_TABLE).set_index("channel").to_dict("index")
    rec = []
    for c, ch in enumerate(CHANNELS):
        t = truth.get(ch, {})
        for param, est, true in (
            ("decay", draws["decay"][:, c], t.get("decay")),
            ("roi", roi[:, c], t.get("roi")),
            ("marginal_roi", mroi[:, c], t.get("marginal_roi_at_run_rate")),
        ):
            rec.append(dict(channel=ch, parameter=param, true_value=true,
                            estimate=est.mean(), estimate_p05=q(est, 0.05),
                            estimate_p95=q(est, 0.95)))
    outputs["MMM_PARAMETER_RECOVERY"] = pd.DataFrame(rec)

    # --- 3. Publish --------------------------------------------------------------------
    for table, frame in outputs.items():
        print(f"  wrote {reader.write(table, frame)}")

    meta = dict(
        model_version=MODEL_VERSION,
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_source=reader.label,
        weeks=n, first_week=str(weeks.min().date()), last_week=str(weeks.max().date()),
        r2=r2, mape=in_mape, holdout_weeks=HOLDOUT_WEEKS, holdout_mape=holdout_mape,
        max_rhat=rhat, draws=len(draws["decay"]),
        run_rate_weeks=RUN_RATE_WEEKS, quarter_weeks=QUARTER_WEEKS,
    )
    print(f"  wrote {reader.write('MMM_MODEL_RUN', pd.DataFrame([meta]))}")
    print(f"Done in {time.time() - t0:.0f}s")
    print(outputs["MMM_CHANNEL_SUMMARY"][["channel", "roi", "marginal_roi", "decay"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
