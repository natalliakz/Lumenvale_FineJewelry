"""Data access for the Lumenvale marketing mix demo: Snowflake first, local files second.

Every table the project uses has the same name in Snowflake and on disk:

* Inputs live in ``data/synthetic-<table>.csv`` and ``LUMENVALE_MMM.PUBLIC.<TABLE>``.
* Model outputs live in ``outputs/<table>.csv`` and the same Snowflake schema.

``MMM_DATA_SOURCE`` chooses the source:

* ``auto`` (default): use Snowflake when credentials are available (Workbench,
  Connect); otherwise, or for a table that has not been loaded yet, use local files.
* ``snowflake``: read and write Snowflake only; fail loudly if anything is missing.
* ``local``: always read and write files (offline).

Set up the Snowflake side with ``snowflake_setup/`` (see its README).

Credentials never live in code:

* **Posit Workbench**: Workbench-managed Snowflake credentials (connection "workbench").
* **Posit Connect**: the viewer's own Snowflake identity via the Posit SDK, so
  row access policies apply per viewer.

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATABASE = os.getenv("MMM_DATABASE", "LUMENVALE_MMM")
SCHEMA = os.getenv("MMM_SCHEMA", "PUBLIC")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "DEFAULT_WH")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "duloftf-posit-software-pbc-dev")

INPUT_TABLES = ("MARKETING_SPEND_WEEKLY", "SALES_WEEKLY", "CONTROLS_WEEKLY")
# The generator's answer key: only exists because the data is synthetic.
TRUTH_TABLE = "SYNTHETIC_TRUE_PARAMETERS"
OUTPUT_TABLES = (
    "MMM_POSTERIOR_DRAWS",
    "MMM_RESPONSE_CURVES",
    "MMM_CONTRIBUTIONS_WEEKLY",
    "MMM_CHANNEL_SUMMARY",
    "MMM_FIT",
    "MMM_PARAMETER_RECOVERY",
    "MMM_MODEL_RUN",
)
SCENARIO_TABLE = "MMM_SAVED_SCENARIOS"
CROSSCHECK_TABLE = "MMM_R_CROSSCHECK"
ALL_TABLES = INPUT_TABLES + (TRUTH_TABLE,) + OUTPUT_TABLES + (SCENARIO_TABLE, CROSSCHECK_TABLE)

# Tables stored locally as JSON rather than CSV.
TRUTH_PATH = DATA_DIR / "synthetic-true_parameters.json"
MODEL_RUN_PATH = OUTPUT_DIR / "model_metadata.json"

DATE_COLUMNS = ("week_start", "saved_at", "checked_at")  # parsed to datetimes on read
# Column types sent to Snowflake (everything else is inferred from pandas).
SNOWFLAKE_DATES = ("week_start", "first_week", "last_week")
SNOWFLAKE_TIMESTAMPS = ("saved_at", "fitted_at", "checked_at")


def data_source() -> str:
    return os.getenv("MMM_DATA_SOURCE", "auto").lower()


def running_on_connect() -> bool:
    return os.getenv("RSTUDIO_PRODUCT") == "CONNECT"


def local_path(table: str) -> Path:
    table = table.upper()
    if table == TRUTH_TABLE:
        return TRUTH_PATH
    if table == "MMM_MODEL_RUN":
        return MODEL_RUN_PATH
    if table in INPUT_TABLES:
        return DATA_DIR / f"synthetic-{table.lower()}.csv"
    return OUTPUT_DIR / f"{table.lower()}.csv"


def read_local(table: str) -> pd.DataFrame:
    """A table from the project folder, in the same shape as its Snowflake version."""
    table = table.upper()
    path = local_path(table)
    if table == TRUTH_TABLE:
        channels = json.loads(path.read_text())["channels"]
        return pd.DataFrame([dict(channel=ch, **v) for ch, v in channels.items()])
    if table == "MMM_MODEL_RUN":
        return _parse_dates(pd.DataFrame([json.loads(path.read_text())]))
    return _parse_dates(pd.read_csv(path))


def ensure_input_data() -> None:
    """Synthetic inputs are not in Git: regenerate them if any file is missing."""
    if not all(local_path(t).exists() for t in INPUT_TABLES + (TRUTH_TABLE,)):
        import runpy

        runpy.run_path(str(DATA_DIR / "generate_data.py"), run_name="__main__")


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------
def connect(user_session_token: str | None = None):
    """Open a Snowflake connection suited to where this code is running."""
    import snowflake.connector

    kwargs = dict(warehouse=WAREHOUSE, database=DATABASE, schema=SCHEMA, login_timeout=15)
    if running_on_connect():
        # Viewer-level credentials: each viewer queries Snowflake as themselves.
        from posit.connect.external.snowflake import PositAuthenticator

        auth = PositAuthenticator(
            local_authenticator="EXTERNALBROWSER", user_session_token=user_session_token
        )
        return snowflake.connector.connect(
            account=ACCOUNT, authenticator=auth.authenticator, token=auth.token, **kwargs
        )
    if os.getenv("SNOWFLAKE_HOME") is not None or (Path.home() / ".snowflake").exists():
        # Posit Workbench managed credentials.
        return snowflake.connector.connect(connection_name="workbench", **kwargs)
    raise RuntimeError("No Snowflake credentials found (not on Workbench or Connect).")


def query(sql: str, con) -> pd.DataFrame:
    """Run SQL in Snowflake and return a pandas DataFrame with lower-case columns."""
    with con.cursor() as cur:
        cur.execute(sql)
        df = cur.fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    return _parse_dates(df)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class TableReader:
    """Reads and writes project tables and remembers where they actually came from."""

    def __init__(self, user_session_token: str | None = None):
        self.source = data_source()
        self._token = user_session_token
        self._con = None
        self.active = "local"
        self.error: str | None = None
        self.local_fallbacks: list[str] = []  # auto mode: tables not found in Snowflake
        if self.source in ("snowflake", "auto"):
            try:
                self._con = connect(user_session_token)
                self.active = "snowflake"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                if self.source == "snowflake":
                    raise
                self.error = str(exc)

    @property
    def label(self) -> str:
        if self.active != "snowflake":
            return "Local files (offline mode)"
        label = f"Snowflake · {DATABASE}.{SCHEMA}"
        if self.local_fallbacks:
            label += f" ({len(self.local_fallbacks)} table(s) from local files)"
        return label

    def origin(self, table: str) -> str:
        """Where a table that has been read came from, e.g. for the app's model badge."""
        table = table.upper()
        if self.active == "snowflake" and table not in self.local_fallbacks:
            return f"Snowflake · {DATABASE}.{SCHEMA}.{table}"
        return f"local file {local_path(table).relative_to(PROJECT_DIR)}"

    def read(self, table: str, sql: str | None = None) -> pd.DataFrame:
        """Read a table. ``sql`` lets Snowflake do the aggregation when connected."""
        table = table.upper()
        if self._con is not None:
            try:
                df = query(sql or f"SELECT * FROM {DATABASE}.{SCHEMA}.{table}", self._con)
            except Exception as exc:  # noqa: BLE001
                # In auto mode a table that has not been created yet falls back to files.
                if self.source == "snowflake" or not _is_missing_table(exc):
                    raise
                df = None
            if df is not None and (len(df) or table == SCENARIO_TABLE):
                return df
            # Missing, or created by schema.sql but not loaded / fitted yet.
            if self.source == "snowflake":
                raise RuntimeError(f"{DATABASE}.{SCHEMA}.{table} is empty. Load snowflake_setup/ "
                                   "or run `uv run python ml/train_model.py`.")
            self.local_fallbacks.append(table)
            self.error = f"{table} is not loaded in Snowflake yet; read from local files."
        if table in INPUT_TABLES or table == TRUTH_TABLE:
            ensure_input_data()
        if not local_path(table).exists():
            if table == SCENARIO_TABLE:
                return pd.DataFrame()
            raise FileNotFoundError(
                f"{local_path(table)} not found. Run `uv run python ml/train_model.py`.")
        return read_local(table)

    def write(self, table: str, df: pd.DataFrame, overwrite: bool = True) -> str:
        """Write a table to Snowflake when connected, and always to a local file."""
        table = table.upper()
        path = local_path(table)
        path.parent.mkdir(exist_ok=True)
        if table == "MMM_MODEL_RUN":
            row = df.iloc[0].to_dict()
            path.write_text(json.dumps({k: _jsonable(v) for k, v in row.items()}, indent=2))
        elif overwrite or not path.exists():
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)
        if self._con is not None:
            write_table(self._con, table, df, overwrite=overwrite)
            return f"{DATABASE}.{SCHEMA}.{table} and {path.name}"
        return str(path.relative_to(PROJECT_DIR))

    def model_run(self) -> dict:
        """The latest model fit's metadata (version, date, fit quality) as a dict."""
        try:
            row = self.read("MMM_MODEL_RUN").iloc[-1].to_dict()
        except FileNotFoundError:
            return {}
        meta = {k: _jsonable(v) for k, v in row.items()}
        for k in ("first_week", "last_week"):  # DATE columns: keep them as YYYY-MM-DD
            if meta.get(k):
                meta[k] = str(meta[k])[:10]
        return meta


def write_table(con, table: str, df: pd.DataFrame, overwrite: bool = True) -> int:
    """Insert a DataFrame into DATABASE.SCHEMA.<table>; overwrite=True deletes first.

    Plain INSERTs (no stage), so the tables and column types come from
    snowflake_setup/schema.sql and a viewer only needs INSERT to save a scenario.
    """
    out = df.copy()
    for col in out.columns:
        if col in SNOWFLAKE_DATES:
            out[col] = pd.to_datetime(out[col]).dt.date
        elif col in SNOWFLAKE_TIMESTAMPS:
            stamps = pd.to_datetime(out[col], utc=True).dt.tz_localize(None)
            out[col] = pd.Series([t.to_pydatetime() for t in stamps], index=out.index, dtype=object)
    out = out.astype(object).where(out.notna(), None)
    rows = [tuple(_jsonable(v) if isinstance(v, np.generic) else v for v in r)
            for r in out.itertuples(index=False)]
    name = f"{DATABASE}.{SCHEMA}.{table.upper()}"
    cols = ", ".join(c.upper() for c in out.columns)
    sql = f"INSERT INTO {name} ({cols}) VALUES ({', '.join(['%s'] * len(out.columns))})"
    with con.cursor() as cur:
        try:
            if overwrite:
                cur.execute(f"DELETE FROM {name}")
            for i in range(0, len(rows), 1000):
                cur.executemany(sql, rows[i:i + 1000])
        except Exception as exc:
            if _is_missing_table(exc):
                raise RuntimeError(f"{name} does not exist. Run snowflake_setup/schema.sql first.") from exc
            raise
    return len(rows)


def _is_missing_table(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "does not exist" in msg or "not authorized" in msg


def _jsonable(v):
    if isinstance(v, (pd.Timestamp, date)):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


def load_model_inputs(reader: TableReader) -> pd.DataFrame:
    """One row per week: total revenue, spend per channel, and controls.

    In Snowflake the pivot and the online/showroom roll-up run in the warehouse.
    """
    if reader.active == "snowflake":
        sales = reader.read("SALES_WEEKLY", f"""
            SELECT week_start, SUM(revenue) AS revenue, SUM(orders) AS orders
            FROM {DATABASE}.{SCHEMA}.SALES_WEEKLY GROUP BY week_start""")
    else:
        sales = (reader.read("SALES_WEEKLY")
                 .groupby("week_start", as_index=False)[["revenue", "orders"]].sum())
    spend = reader.read("MARKETING_SPEND_WEEKLY").pivot(
        index="week_start", columns="channel", values="spend").reset_index()
    spend.columns.name = None
    controls = reader.read("CONTROLS_WEEKLY")
    return (sales.merge(spend, on="week_start").merge(controls, on="week_start")
            .sort_values("week_start").reset_index(drop=True))
