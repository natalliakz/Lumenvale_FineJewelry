"""Generate synthetic weekly marketing, sales and control data for Lumenvale Jewelers.

Lumenvale Jewelers is a FICTIONAL omnichannel fine-jewelry retailer. Every number
produced here is simulated. The simulation is a known marketing mix: each channel
has a true carryover (adstock) rate and a true saturation curve, which are saved to
``synthetic-true_parameters.json`` so the EDA can check the model recovered them.

The "aha" is baked in on purpose:
* Paid Search gets the most money and is deep into saturation (low marginal ROI).
* Connected TV gets little money and is still on the steep part of its curve.

Run:  uv run python data/generate_data.py

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260924
DATA_DIR = Path(__file__).resolve().parent
N_WEEKS = 156  # three years of Mondays
FIRST_WEEK = date(2023, 1, 2)
MAX_LAG = 13  # weeks of carryover the adstock looks back

# True media parameters, in dollars per week.
#   base_spend  typical weekly spend
#   decay       share of this week's effect that carries into next week
#   k           saturation scale: the curve is half-way to its ceiling at x = 1.1 * k
#   beta        ceiling: the most weekly revenue the channel can ever drive
#   cpm         cost per thousand impressions, used only to derive impressions
CHANNELS = {
    "Paid Search":  dict(base_spend=150_000, decay=0.05, k=40_000,  beta=700_000, cpm=38.0),
    "Connected TV": dict(base_spend=30_000,  decay=0.75, k=120_000, beta=900_000, cpm=32.0),
    "Paid Social":  dict(base_spend=90_000,  decay=0.30, k=70_000,  beta=380_000, cpm=9.5),
    "Display":      dict(base_spend=50_000,  decay=0.30, k=45_000,  beta=120_000, cpm=4.0),
    "Affiliate":    dict(base_spend=45_000,  decay=0.10, k=50_000,  beta=200_000, cpm=12.0),
    "Email / CRM":  dict(base_spend=12_000,  decay=0.05, k=5_000,   beta=60_000,  cpm=1.5),
    "Direct Mail":  dict(base_spend=40_000,  decay=0.40, k=60_000,  beta=150_000, cpm=650.0),
    "Podcast":      dict(base_spend=18_000,  decay=0.65, k=60_000,  beta=250_000, cpm=25.0),
}

# Baseline (non-media) revenue and control effects, in dollars per week.
BASELINE = 1_600_000
TREND_PER_YEAR = 0.07  # organic growth, as a share of BASELINE
PROMO_LIFT = 0.07 * BASELINE
HOLIDAY_LIFT = 0.12 * BASELINE
CCI_EFFECT = 0.006 * BASELINE  # per index point away from 100
DPI_EFFECT = -0.004 * BASELINE  # per index point away from 100: pricier stones, fewer sales
NOISE_SD = 0.025  # share of expected revenue


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = MAX_LAG) -> np.ndarray:
    """Carry spend forward with weights decay**lag, normalised to sum to one.

    Normalising means a steady weekly spend of S has a steady adstock of S, so the
    saturation curve can be read directly in "dollars per week".
    """
    weights = decay ** np.arange(max_lag)
    weights /= weights.sum()
    out = np.zeros_like(x, dtype=float)
    for lag, w in enumerate(weights):
        out[lag:] += w * x[: len(x) - lag]
    return out


def logistic_saturation(x: np.ndarray, k: float) -> np.ndarray:
    """0 at zero spend, rising towards 1 as x grows past k."""
    u = x / k
    return (1 - np.exp(-u)) / (1 + np.exp(-u))


def seasonal_bump(doy: np.ndarray, center: float, width: float) -> np.ndarray:
    """A smooth bump on the day-of-year circle."""
    d = np.minimum(np.abs(doy - center), 365.25 - np.abs(doy - center))
    return np.exp(-0.5 * (d / width) ** 2)


def main() -> None:
    rng = np.random.default_rng(SEED)
    weeks = [FIRST_WEEK + timedelta(weeks=i) for i in range(N_WEEKS)]
    doy = np.array([w.timetuple().tm_yday + 3 for w in weeks], dtype=float)  # mid-week
    t_years = np.arange(N_WEEKS) / 52.0

    # --- Seasonality (share of BASELINE) ------------------------------------
    # Q4 holiday ramp, a sharp December engagement spike, and summer wedding season.
    season = (
        0.22 * seasonal_bump(doy, 340, 22)  # Q4 holiday shopping
        + 0.30 * seasonal_bump(doy, 356, 10)  # December engagement / proposal spike
        + 0.12 * seasonal_bump(doy, 190, 35)  # summer wedding season (bands)
        - 0.10 * seasonal_bump(doy, 20, 15)  # January lull
    )

    # --- Controls -------------------------------------------------------------
    def near(month: int, day: int, window: int = 3) -> np.ndarray:
        out = []
        for w in weeks:
            target = date(w.year, month, day)
            out.append(abs((w + timedelta(days=3) - target).days) <= window)
        return np.array(out)

    def mothers_day(year: int) -> date:
        may1 = date(year, 5, 1)
        first_sunday = may1 + timedelta(days=(6 - may1.weekday()) % 7)
        return first_sunday + timedelta(weeks=1)

    def thanksgiving(year: int) -> date:
        nov1 = date(year, 11, 1)
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        return first_thu + timedelta(weeks=3)

    holiday = (
        near(2, 14)  # Valentine's Day
        | np.array([abs((w + timedelta(days=3) - mothers_day(w.year)).days) <= 3 for w in weeks])
        | np.array([abs((w + timedelta(days=3) - thanksgiving(w.year)).days) <= 3 for w in weeks])
        | near(12, 22, 4)  # Christmas week
    ).astype(int)

    promo = (rng.random(N_WEEKS) < 0.10).astype(int)
    promo[holiday == 1] = 1  # every holiday week runs a promotion
    promo[(doy > 320) & (rng.random(N_WEEKS) < 0.5)] = 1  # extra Q4 promos

    cci = 100 + np.cumsum(rng.normal(0, 0.9, N_WEEKS))
    cci = 100 + (cci - cci.mean())  # consumer confidence, centred on 100
    dpi = 108 - 11 * t_years + rng.normal(0, 1.2, N_WEEKS)  # stone prices drift down

    # --- Media spend ------------------------------------------------------------
    # Budgets follow the calendar (heavier in Q4 and around Valentine's) with noise.
    flighting = 1 + 0.35 * seasonal_bump(doy, 340, 25) + 0.20 * seasonal_bump(doy, 40, 10)
    spend = {}
    for name, p in CHANNELS.items():
        s = p["base_spend"] * flighting * rng.lognormal(0, 0.30, N_WEEKS)
        if name == "Connected TV":
            # TV runs in flights: 6 weeks on, 3 weeks off, plus a 2x test in autumn 2024.
            on = (np.arange(N_WEEKS) % 9) < 6
            s = np.where(on, s * 1.35, 0.0)
            test = (np.array(weeks) >= date(2024, 9, 2)) & (np.array(weeks) <= date(2024, 10, 28))
            s[test] *= 2.2
        if name == "Podcast":
            on = (np.arange(N_WEEKS) % 8) < 5
            s = np.where(on, s * 1.4, 0.0)
        if name == "Paid Search":
            # A four-week budget freeze in spring 2024 exposes the curve at low spend.
            freeze = (np.array(weeks) >= date(2024, 4, 1)) & (np.array(weeks) <= date(2024, 4, 22))
            s[freeze] *= 0.35
        if name == "Direct Mail":
            s *= np.where((np.arange(N_WEEKS) % 4) == 0, 2.2, 0.6)  # monthly drops
        spend[name] = np.round(s, 2)

    # --- Revenue ------------------------------------------------------------------
    contributions = {
        name: p["beta"] * logistic_saturation(geometric_adstock(spend[name], p["decay"]), p["k"])
        for name, p in CHANNELS.items()
    }
    media = np.sum(list(contributions.values()), axis=0)
    baseline = (
        BASELINE * (1 + TREND_PER_YEAR * t_years + season)
        + PROMO_LIFT * promo
        + HOLIDAY_LIFT * holiday
        + CCI_EFFECT * (cci - 100)
        + DPI_EFFECT * (dpi - 100)
    )
    expected = baseline + media
    revenue = expected * (1 + rng.normal(0, NOISE_SD, N_WEEKS))

    # Split into online and showroom; showroom orders carry a higher ticket.
    online_share = np.clip(0.60 + 0.04 * t_years + rng.normal(0, 0.02, N_WEEKS), 0.5, 0.75)
    aov_lift = 1 + 0.12 * seasonal_bump(doy, 356, 10)  # engagement rings lift December AOV
    aov = {
        "Online": 2_450 * aov_lift * rng.normal(1, 0.03, N_WEEKS),
        "Showroom": 3_700 * aov_lift * rng.normal(1, 0.03, N_WEEKS),
    }
    share = {"Online": online_share, "Showroom": 1 - online_share}

    # --- Assemble tables -------------------------------------------------------------
    spend_rows = []
    for name, p in CHANNELS.items():
        impressions = spend[name] / p["cpm"] * 1000 * rng.normal(1, 0.05, N_WEEKS)
        for i, w in enumerate(weeks):
            spend_rows.append((w, name, float(spend[name][i]), int(max(impressions[i], 0))))
    spend_df = pl.DataFrame(
        spend_rows, schema=["week_start", "channel", "spend", "impressions"], orient="row"
    )

    sales_rows = []
    for ch in ("Online", "Showroom"):
        rev = revenue * share[ch]
        orders = np.round(rev / aov[ch]).astype(int)
        for i, w in enumerate(weeks):
            sales_rows.append((w, ch, int(orders[i]), round(float(rev[i]), 2),
                               round(float(rev[i] / orders[i]), 2)))
    sales_df = pl.DataFrame(
        sales_rows,
        schema=["week_start", "sales_channel", "orders", "revenue", "avg_order_value"],
        orient="row",
    )

    controls_df = pl.DataFrame({
        "week_start": weeks,
        "promo_flag": promo,
        "holiday_flag": holiday,
        "consumer_confidence_index": np.round(cci, 2),
        "diamond_price_index": np.round(dpi, 2),
    })

    spend_df.write_csv(DATA_DIR / "synthetic-marketing_spend_weekly.csv")
    sales_df.write_csv(DATA_DIR / "synthetic-sales_weekly.csv")
    controls_df.write_csv(DATA_DIR / "synthetic-controls_weekly.csv")

    # --- Ground truth for the EDA ------------------------------------------------------
    truth = {"channels": {}, "note": "Synthetic ground truth. Dollar units are per week."}
    for name, p in CHANNELS.items():
        s = spend[name]
        run_rate = float(s[-52:].mean())  # trailing 52-week average weekly spend
        u = run_rate / p["k"]
        slope = p["beta"] / p["k"] * 2 * np.exp(-u) / (1 + np.exp(-u)) ** 2
        truth["channels"][name] = {
            "decay": p["decay"],
            "k": p["k"],
            "beta": p["beta"],
            "total_spend": float(s.sum()),
            "total_contribution": float(contributions[name].sum()),
            "roi": float(contributions[name].sum() / s.sum()),
            "marginal_roi_at_run_rate": float(slope),
        }
    (DATA_DIR / "synthetic-true_parameters.json").write_text(json.dumps(truth, indent=2))

    print(f"Wrote {len(spend_df):,} spend rows, {len(sales_df):,} sales rows, "
          f"{len(controls_df):,} control rows to {DATA_DIR}")
    for name, c in truth["channels"].items():
        print(f"  {name:13s} ROI {c['roi']:.2f}  marginal ROI {c['marginal_roi_at_run_rate']:.2f}")


if __name__ == "__main__":
    main()
