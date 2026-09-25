"""The app's side of the Snowflake Model Registry: find the model and call it.

``ml/register_model.py`` logs the Channel Investment Planner model to the registry
as ``LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM`` (one version per fit, e.g. ``V3_2``) and
makes it the default version. The Streamlit app then:

1. asks the registry for the default version (``SHOW MODELS`` / ``SHOW VERSIONS``),
   which the header badge shows, and
2. runs the scenario on that version in a Snowflake warehouse:

       SELECT MODEL(LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM, V3_2)!RECOMMEND(...)

Plain SQL over the app's existing Snowflake connection, so the app does not need
snowflake-ml-python, and on Connect each viewer calls the model as themselves.

``MMM_SCORING`` chooses where scenarios are scored:

* ``auto`` (default): the registered model when it is available, else locally from
  ``MMM_POSTERIOR_DRAWS`` (same math, same numbers).
* ``registry``: the registered model only; fail loudly if it is missing.
* ``local``: always locally.

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from mmm_data import DATABASE, SCHEMA, TableReader

MODEL_NAME = os.getenv("MMM_MODEL_NAME", "LUMENVALE_MMM")
MODEL_FQN = f"{DATABASE}.{SCHEMA}.{MODEL_NAME}"


def scoring_mode() -> str:
    return os.getenv("MMM_SCORING", "auto").lower()


def version_name(model_version: str) -> str:
    """'v3.2' -> 'V3_2': registry version names must be SQL identifiers."""
    return model_version.upper().replace(".", "_").replace("-", "_")


def display_version(name: str) -> str:
    """'V3_2' -> 'v3.2', the way the model version is written everywhere else."""
    return name.lower().replace("_", ".")


@dataclass
class RegisteredModel:
    fqn: str
    version: str  # registry version name, e.g. V3_2
    metrics: dict = field(default_factory=dict)
    comment: str = ""
    created_on: str = ""

    @property
    def model_version(self) -> str:
        return str(self.metrics.get("model_version") or display_version(self.version))

    @property
    def label(self) -> str:
        return f"Snowflake Model Registry · {self.fqn} · {self.version} (default)"


def default_model(reader: TableReader) -> RegisteredModel | None:
    """The registered model's default version, or None if it is not registered (auto mode)."""
    mode = scoring_mode()
    if mode == "local":
        return None
    try:
        if reader.active != "snowflake":
            raise RuntimeError("not connected to Snowflake")
        models = reader.rows(f"SHOW MODELS LIKE '{MODEL_NAME}' IN SCHEMA {DATABASE}.{SCHEMA}")
        if not models:
            raise RuntimeError(f"{MODEL_FQN} is not registered. Run `uv run python ml/register_model.py`.")
        default = models[0]["default_version_name"]
        versions = reader.rows(f"SHOW VERSIONS LIKE '{default}' IN MODEL {MODEL_FQN}")
        row = versions[0] if versions else {}
        metadata = json.loads(row.get("metadata") or "{}")
        return RegisteredModel(fqn=MODEL_FQN, version=default,
                               metrics=metadata.get("metrics") or {},
                               comment=row.get("comment") or "",
                               created_on=str(row.get("created_on") or "")[:10])
    except Exception:
        if mode == "registry":
            raise
        return None


def _literal(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NULL::FLOAT"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return f"{float(v)!r}::FLOAT"


def call(reader: TableReader, model: RegisteredModel, method: str, inputs: dict) -> dict:
    """Run one method of the registered model in Snowflake on one row of inputs.

    ``inputs`` maps the method's input columns, in signature order, to values.
    """
    args = ", ".join(f"{_literal(v)} AS {c}" for c, v in inputs.items())
    sql = (f"SELECT MODEL({model.fqn}, {model.version})!{method.upper()}({', '.join(inputs)}) "
           f"AS OUT FROM (SELECT {args})")
    out = reader.rows(sql)[0]["out"]
    return json.loads(out) if isinstance(out, str) else out
