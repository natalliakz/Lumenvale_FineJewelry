"""Scenario math for the Channel Investment Planner.

Everything here works on precomputed posterior draws of each channel's response
curve, so projecting or optimising a plan takes milliseconds. No model is refit.

A plan is a quarterly budget per channel. It is spread evenly over the quarter's
13 weeks and evaluated at steady state: weekly revenue = beta * sat(weekly spend / k).

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

QUARTER_WEEKS = 13


@dataclass
class Curves:
    """Posterior draws of every channel's response curve: arrays of shape (draw, channel)."""

    channels: list[str]
    k: np.ndarray
    beta: np.ndarray

    @classmethod
    def from_draws(cls, draws: pd.DataFrame, channels: list[str]) -> "Curves":
        wide = lambda col: draws.pivot(index="draw", columns="channel", values=col)[channels]
        return cls(channels, wide("k").to_numpy(), wide("beta").to_numpy())

    def revenue(self, quarterly: np.ndarray) -> np.ndarray:
        """Quarterly media revenue per draw and channel for a quarterly budget per channel."""
        u = np.asarray(quarterly, float)[None, :] / QUARTER_WEEKS / self.k
        return QUARTER_WEEKS * self.beta * (1 - np.exp(-u)) / (1 + np.exp(-u))

    def mean_revenue_and_grad(self, quarterly: np.ndarray) -> tuple[float, np.ndarray]:
        """Posterior-mean total revenue of a plan, and its gradient (revenue per extra $)."""
        u = np.asarray(quarterly, float)[None, :] / QUARTER_WEEKS / self.k
        e = np.exp(-u)
        rev = QUARTER_WEEKS * self.beta * (1 - e) / (1 + e)
        grad = self.beta / self.k * 2 * e / (1 + e) ** 2
        return float(rev.sum(1).mean()), grad.mean(0)


def project(curves: Curves, plan: np.ndarray, baseline_plan: np.ndarray) -> dict:
    """Projected revenue and ROI of a plan, with 90% intervals and the change vs baseline.

    The change vs baseline is computed draw by draw, so it is tighter than the
    difference of the two intervals.
    """
    rev = curves.revenue(plan).sum(1)
    base = curves.revenue(baseline_plan).sum(1)
    spend = float(np.sum(plan))
    roi = rev / spend if spend > 0 else np.zeros_like(rev)
    delta = rev - base

    def band(a):
        return float(a.mean()), float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95))

    return dict(
        spend=spend,
        revenue=band(rev),
        roi=band(roi),
        delta=band(delta),
        by_channel=curves.revenue(plan).mean(0),
    )


def optimize(curves: Curves, total: float, lower: np.ndarray, upper: np.ndarray,
             start: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Maximise posterior-mean revenue for a fixed total budget within per-channel bounds.

    Returns the plan and a status message. If the bounds make the total impossible,
    the nearest feasible total is used and the message says so.
    """
    lower = np.minimum(np.asarray(lower, float), np.asarray(upper, float))
    upper = np.asarray(upper, float)
    note = "Optimised within your constraints."
    if total < lower.sum():
        total, note = lower.sum(), "Minimums exceed the budget: using the sum of minimums."
    elif total > upper.sum():
        total, note = upper.sum(), "Maximums are below the budget: every channel is at its cap."
    if np.isclose(lower.sum(), upper.sum()) or total == upper.sum():
        return upper.copy() if total == upper.sum() else lower.copy(), note

    x0 = np.clip(start if start is not None else upper * 0 + total / len(upper), lower, upper)
    x0 = lower + (x0 - lower) * (total - lower.sum()) / max((x0 - lower).sum(), 1e-9)
    x0 = np.clip(x0, lower, upper)

    scale = max(total, 1.0)  # work in units of the total budget for a well-conditioned solve

    def objective(z):
        value, grad = curves.mean_revenue_and_grad(z * scale)
        return -value / scale, -grad

    res = minimize(
        objective, x0 / scale, jac=True, method="SLSQP",
        bounds=list(zip(lower / scale, upper / scale)),
        constraints=[{"type": "eq", "fun": lambda z: z.sum() - total / scale,
                      "jac": lambda z: np.ones_like(z)}],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    plan = np.clip(res.x * scale, lower, upper)
    return plan, note if res.success else f"{note} (solver: {res.message})"
