# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "snowflake-ml-python>=1.20",
#     "snowflake-connector-python[pandas]",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
"""Log the fitted marketing mix model to the Snowflake Model Registry.

Run after ``ml/train_model.py``, on Posit Workbench (Workbench-managed Snowflake
credentials):

    uv run ml/register_model.py

The script header above makes uv run it on Python 3.12 with snowflake-ml-python,
separately from the project's Python 3.14 environment. It must match the Python the
warehouse runs the model on: the registry stores the model class as bytecode.

Steps
1. Read the fit from LUMENVALE_MMM.PUBLIC: MMM_POSTERIOR_DRAWS (the model),
   MMM_CHANNEL_SUMMARY (the current plan) and MMM_MODEL_RUN (version, fit quality).
2. Log it as a CustomModel (ml/mmm_custom_model.py) named LUMENVALE_MMM, version
   V3_2 for model v3.2, with the fit quality as metrics, runnable in a warehouse.
3. Make that version the default: the Streamlit app always runs the default.
4. Call RECOMMEND in Snowflake and check it matches the local calculation.

A version is logged once. To publish a new one, refit with a new MODEL_VERSION in
ml/train_model.py; to roll back, set an older version as the default:
    ALTER MODEL LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM SET DEFAULT_VERSION = V3_1;

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "ml"))

from mmm_data import DATABASE, SCHEMA, TableReader  # noqa: E402
from mmm_registry import MODEL_FQN, MODEL_NAME, version_name  # noqa: E402
from model_utils import CHANNELS  # noqa: E402
from planner import (  # noqa: E402
    Curves, current_plan, plan_columns, recommend, recommend_columns,
)

# The Python the model runs on in the warehouse (the version this script runs on).
WAREHOUSE_PYTHON = (3, 12)
# Loose ranges that the Snowflake Anaconda channel has; the model uses basic APIs only.
DEPENDENCIES = ["numpy>=1.26,<3", "pandas>=2.0,<3", "scipy>=1.11"]
METRICS = ("model_version", "fitted_at", "weeks", "first_week", "last_week", "r2", "mape",
           "holdout_weeks", "holdout_mape", "max_rhat", "draws")
# The demo's headline question: +$200K with Connected TV capped at +$60K.
SMOKE_TEST = dict(extra=200_000.0, caps={"Connected TV": 60_000.0})


def main() -> None:
    if sys.version_info[:2] != WAREHOUSE_PYTHON:
        sys.exit("Run this with `uv run ml/register_model.py` (Python 3.12, see the header).")
    from snowflake.ml.model import custom_model, model_signature
    from snowflake.ml.registry import Registry
    from snowflake.snowpark import Session

    from mmm_custom_model import DRAWS_ARTIFACT, SPEC_ARTIFACT, ChannelInvestmentModel

    reader = TableReader()
    if reader.active != "snowflake":
        sys.exit(f"Needs Snowflake (the registry lives there): {reader.error}")

    # --- 1. The fit, as the app would read it ---------------------------------------
    draws = reader.read("MMM_POSTERIOR_DRAWS")
    summary = reader.read("MMM_CHANNEL_SUMMARY").set_index("channel").loc[CHANNELS]
    meta = reader.model_run()
    if reader.local_fallbacks:
        sys.exit(f"{', '.join(reader.local_fallbacks)} not in Snowflake yet: "
                 "run ml/train_model.py or load snowflake_setup/ first.")
    version = version_name(meta["model_version"])
    current = current_plan(summary["current_quarter_budget"].to_numpy())
    print(f"Model {meta['model_version']} (fit {meta['fitted_at'][:10]}) -> {MODEL_FQN} {version}")

    # --- 2. Log it ------------------------------------------------------------------
    registry = Registry(Session.builder.configs({"connection": reader._con}).create(),
                        database_name=DATABASE, schema_name=SCHEMA)
    try:
        model = registry.get_model(MODEL_NAME)
    except ValueError:  # not registered yet
        model = None
    if model is not None and version in list(model.show_versions()["name"]):
        print(f"  {version} is already registered; not logging it again.")
        mv = model.version(version)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            draws_path, spec_path = Path(tmp, "posterior_draws.csv"), Path(tmp, "model_spec.json")
            draws[["draw", "channel", "decay", "k", "beta"]].to_csv(draws_path, index=False)
            spec_path.write_text(json.dumps(dict(
                channels=CHANNELS, current_plan=current.tolist(),
                model_version=meta["model_version"], fitted_at=meta["fitted_at"])))
            ctx = custom_model.ModelContext(artifacts={DRAWS_ARTIFACT: str(draws_path),
                                                       SPEC_ARTIFACT: str(spec_path)})
            local = ChannelInvestmentModel(ctx)
            plans, questions = sample_inputs(current)
            signatures = {
                "predict": model_signature.infer_signature(plans, local.predict(plans)),
                "recommend": model_signature.infer_signature(questions, local.recommend(questions)),
            }
            mv = registry.log_model(
                local,
                model_name=MODEL_NAME,
                version_name=version,
                comment=(f"Channel Investment Planner, marketing mix model {meta['model_version']}: "
                         "Bayesian MMM (PyMC), posterior draws of each channel's response curve. "
                         "Synthetic demo data (Lumenvale Jewelers, fictional)."),
                metrics={k: meta[k] for k in METRICS if k in meta},
                conda_dependencies=DEPENDENCIES,
                target_platforms=["WAREHOUSE"],
                signatures=signatures,
                code_paths=[str(PROJECT_DIR / "planner.py"),
                            str(PROJECT_DIR / "ml" / "mmm_custom_model.py")],
            )
            print(f"  logged {MODEL_FQN} version {version}")

    # --- 3. Default version ------------------------------------------------------------
    model = registry.get_model(MODEL_NAME)
    model.default = version
    print(f"  default version: {model.default.version_name}")

    # --- 4. Smoke test: Snowflake vs local --------------------------------------------
    caps = np.array([SMOKE_TEST["caps"].get(ch, np.nan) for ch in CHANNELS])
    question = pd.DataFrame([[SMOKE_TEST["extra"], *caps]], columns=recommend_columns(CHANNELS))
    remote = mv.run(question, function_name="recommend").iloc[0]
    expected = recommend(Curves.from_draws(draws, CHANNELS), current, SMOKE_TEST["extra"], caps)
    got = np.array([remote[f"ADD_{c}"] for c in plan_columns(CHANNELS)], float)
    if not np.allclose(got, expected["add"]):
        sys.exit(f"RECOMMEND in Snowflake gave {got}, but locally {expected['add']}.")
    print("  RECOMMEND in Snowflake: +$200K -> "
          + ", ".join(f"{ch} +${a / 1e3:.0f}K" for ch, a in zip(CHANNELS, got) if a)
          + f"; incremental revenue ${remote['INCREMENTAL'] / 1e3:,.0f}K (matches local)")


def sample_inputs(current: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example rows for each method; they fix the column names and types of its signature."""
    plans = pd.DataFrame([current, current * 1.1], columns=plan_columns(CHANNELS))
    questions = pd.DataFrame(
        [[200_000.0, *[60_000.0 if ch == "Connected TV" else 200_000.0 for ch in CHANNELS]],
         [500_000.0, *[500_000.0] * len(CHANNELS)]],
        columns=recommend_columns(CHANNELS))
    return plans, questions


if __name__ == "__main__":
    main()
