"""Bayesian marketing mix model for Lumenvale Jewelers, written directly in PyMC.

Weekly revenue = intercept + trend + seasonality + controls + sum over channels of
    beta_c * saturation(adstock(spend_c, decay_c), k_c) + noise

* Adstock: geometric carryover, weights decay**lag normalised to sum to one, so a
  steady spend of $S per week has a steady adstock of $S per week.
* Saturation: logistic, (1 - exp(-x/k)) / (1 + exp(-x/k)), rising from 0 towards 1.
* beta_c is the most weekly revenue the channel can drive; k_c sets how fast it gets there.

This is the same structure PyMC-Marketing uses (GeometricAdstock + LogisticSaturation).
See posit-README.md for why it is written out by hand here.

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed display order: also the colour order used in every chart.
CHANNELS = [
    "Paid Search", "Connected TV", "Paid Social", "Display",
    "Affiliate", "Email / CRM", "Direct Mail", "Podcast",
]
CONTROLS = ["promo_flag", "holiday_flag", "consumer_confidence_index", "diamond_price_index"]
MAX_LAG = 13
FOURIER_ORDER = 7


def lagged(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Stack x shifted by 0..max_lag-1 weeks: shape (lag, time, channel)."""
    out = np.zeros((max_lag, *x.shape))
    for lag in range(max_lag):
        out[lag, lag:] = x[: x.shape[0] - lag]
    return out


def design(df: pd.DataFrame) -> dict:
    """Turn the weekly model table into scaled arrays for the model."""
    spend = df[CHANNELS].to_numpy(float)
    spend_scale = spend.max(axis=0)
    week = pd.to_datetime(df["week_start"])
    doy = week.dt.dayofyear.to_numpy() + 3
    fourier = np.column_stack(
        [f(2 * np.pi * k * doy / 365.25) for k in range(1, FOURIER_ORDER + 1) for f in (np.sin, np.cos)]
    )
    ctrl = df[CONTROLS].to_numpy(float)
    ctrl_mean = ctrl.mean(axis=0)
    ctrl_sd = ctrl.std(axis=0)
    ctrl_mean[:2], ctrl_sd[:2] = 0.0, 1.0  # leave the 0/1 flags as they are
    y = df["revenue"].to_numpy(float)
    y_scale = y.mean()
    return dict(
        x_lag=lagged(spend / spend_scale),
        spend=spend,
        spend_scale=spend_scale,
        fourier=fourier,
        controls=(ctrl - ctrl_mean) / ctrl_sd,
        trend=np.arange(len(df)) / len(df),
        y=y / y_scale,
        y_scale=y_scale,
    )


def build_model(d: dict, n_train: int | None = None):
    """The PyMC model. ``n_train`` fits on the first n weeks only (for back-testing)."""
    import pymc as pm
    import pytensor.tensor as pt

    n = n_train or len(d["y"])
    coords = {
        "channel": CHANNELS,
        "fourier": [f"f{i}" for i in range(d["fourier"].shape[1])],
        "control": CONTROLS,
    }
    with pm.Model(coords=coords) as model:
        decay = pm.Beta("decay", 2, 2, dims="channel")
        lam = pm.Gamma("lam", 3, 1, dims="channel")  # saturation rate on scaled spend
        # Ceiling on weekly revenue, as a share of mean revenue. The prior scale is
        # 5x the channel's largest weekly spend: generous, but it keeps small
        # channels from soaking up seasonality they cannot explain.
        beta = pm.HalfNormal("beta", 5 * d["spend_scale"] / d["y_scale"], dims="channel")

        weights = decay[None, :] ** pt.arange(MAX_LAG)[:, None]
        weights = weights / weights.sum(axis=0)
        adstock = (d["x_lag"][:, :n, :] * weights[:, None, :]).sum(axis=0)
        media = beta * (1 - pt.exp(-lam * adstock)) / (1 + pt.exp(-lam * adstock))

        intercept = pm.Normal("intercept", 0.5, 0.3)
        trend = pm.Normal("trend", 0, 0.2)
        season = pm.Normal("season", 0, 0.1, dims="fourier")
        gamma = pm.Normal("gamma", 0, 0.1, dims="control")
        sigma = pm.HalfNormal("sigma", 0.05)

        mu = (
            intercept
            + trend * d["trend"][:n]
            + pt.dot(d["fourier"][:n], season)
            + pt.dot(d["controls"][:n], gamma)
            + media.sum(axis=1)
        )
        pm.Normal("y", mu, sigma, observed=d["y"][:n])
    return model


# ---------------------------------------------------------------------------
# Post-processing in plain numpy, in dollars
# ---------------------------------------------------------------------------
def draws_in_dollars(idata, d: dict, n_draws: int = 400, seed: int = 1) -> dict:
    """Flatten chains, thin to n_draws, and convert channel parameters to dollars."""
    post = idata.posterior
    flat = {v: np.asarray(post[v].values).reshape(-1, *post[v].shape[2:])
            for v in ("decay", "lam", "beta", "intercept", "trend", "season", "gamma", "sigma")}
    total = flat["decay"].shape[0]
    idx = np.random.default_rng(seed).choice(total, size=min(n_draws, total), replace=False)
    out = {k: v[idx] for k, v in flat.items()}
    out["k_dollars"] = d["spend_scale"] / out["lam"]  # saturation scale, $ per week
    out["beta_dollars"] = out["beta"] * d["y_scale"]  # revenue ceiling, $ per week
    return out


def adstock_dollars(spend: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """Adstock for every draw: spend (time, channel), decay (draw, channel) -> (draw, time, channel)."""
    weights = decay[:, None, :] ** np.arange(MAX_LAG)[None, :, None]
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ltc,dlc->dtc", lagged(spend), weights)


def saturation(x, k, beta):
    """Weekly revenue from adstocked weekly spend x, for saturation scale k and ceiling beta."""
    u = x / k
    return beta * (1 - np.exp(-u)) / (1 + np.exp(-u))


def marginal(x, k, beta):
    """Revenue from the next dollar of weekly spend: the slope of the saturation curve."""
    e = np.exp(-x / k)
    return beta / k * 2 * e / (1 + e) ** 2


def media_contributions(draws: dict, d: dict) -> np.ndarray:
    """Weekly revenue per channel for every draw: (draw, time, channel), in dollars."""
    ad = adstock_dollars(d["spend"], draws["decay"])
    return saturation(ad, draws["k_dollars"][:, None, :], draws["beta_dollars"][:, None, :])


def predicted_revenue(draws: dict, d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expected weekly revenue per draw, and the media part of it, both in dollars."""
    base = (
        draws["intercept"][:, None]
        + draws["trend"][:, None] * d["trend"][None, :]
        + draws["season"] @ d["fourier"].T
        + draws["gamma"] @ d["controls"].T
    ) * d["y_scale"]
    media = media_contributions(draws, d)
    return base + media.sum(axis=2), media
