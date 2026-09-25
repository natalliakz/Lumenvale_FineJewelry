"""The Channel Investment Planner model, as registered in the Snowflake Model Registry.

A snowflake-ml ``CustomModel`` that wraps the marketing mix model's posterior draws
(``MMM_POSTERIOR_DRAWS``) and the scenario math in ``planner.py``. Registered by
``ml/register_model.py`` as ``LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM`` and run in a
Snowflake warehouse. It has two methods:

* ``PREDICT``: one row per plan (quarterly $ per channel) gives projected media
  revenue, and the change vs the current plan, with 90% intervals.
* ``RECOMMEND``: one row per question ("where should +$X go, with these caps?")
  gives the split of the extra budget and the projected incremental revenue.

Channel names become column names such as ``PAID_SEARCH`` and ``EMAIL_CRM``
(see ``planner.feature_name``). In SQL, for example:

    WITH mv AS MODEL LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM VERSION V3_2
    SELECT mv!RECOMMEND(200000, NULL, 60000, NULL, NULL, NULL, NULL, NULL, NULL);

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

# No `from __future__ import annotations`: snowflake-ml checks that the methods are
# annotated with the pandas.DataFrame class itself.
import json

import numpy as np
import pandas as pd
from snowflake.ml.model import custom_model

from planner import Curves, plan_columns, project, recommend, recommend_columns

# Files the model carries (its ModelContext artifacts).
DRAWS_ARTIFACT = "posterior_draws"
SPEC_ARTIFACT = "model_spec"


def band(prefix: str, values: tuple[float, float, float]) -> dict:
    mean, p05, p95 = values
    return {prefix: mean, f"{prefix}_P05": p05, f"{prefix}_P95": p95}


class ChannelInvestmentModel(custom_model.CustomModel):
    """Posterior draws of each channel's response curve, plus the current plan."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        spec = json.loads(open(context.path(SPEC_ARTIFACT)).read())
        self.channels: list[str] = spec["channels"]
        self.current = np.asarray(spec["current_plan"], float)
        self.model_version: str = spec["model_version"]
        draws = pd.read_csv(context.path(DRAWS_ARTIFACT))
        self.curves = Curves.from_draws(draws, self.channels)

    @custom_model.inference_api
    def predict(self, plans: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for plan in plans[plan_columns(self.channels)].to_numpy(float):
            proj = project(self.curves, plan, self.current)
            rows.append(dict(SPEND=proj["spend"], **band("REVENUE", proj["revenue"]),
                             **band("INCREMENTAL", proj["delta"]), ROAS=proj["roi"][0],
                             MODEL_VERSION=self.model_version))
        return pd.DataFrame(rows)

    @custom_model.inference_api
    def recommend(self, questions: pd.DataFrame) -> pd.DataFrame:
        caps = questions[recommend_columns(self.channels)[1:]].to_numpy(float)
        rows = []
        for extra, cap in zip(questions["EXTRA_BUDGET"].to_numpy(float), caps):
            res = recommend(self.curves, self.current, float(extra), cap)
            spent = float(res["add"].sum())
            delta = res["proj"]["delta"]
            rows.append(dict(
                **{f"ADD_{c}": float(a) for c, a in zip(plan_columns(self.channels), res["add"])},
                **band("INCREMENTAL", delta), ROAS=delta[0] / spent if spent else 0.0,
                NOTE=res["note"], MODEL_VERSION=self.model_version))
        return pd.DataFrame(rows)
