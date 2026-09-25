# Snowflake setup: `LUMENVALE_MMM.PUBLIC`

> This project contains synthetic data and analysis created for demonstration purposes only.

Everything the demo reads and writes lives in its own Snowflake database,
`LUMENVALE_MMM`, in the default `PUBLIC` schema. This folder has the schema and **all the data, ready to load**, so the whole demo (model, app, EDA and
R reports) runs from Snowflake.

## What gets created

| Table | Rows | Filled by | Read by |
|---|---:|---|---|
| `MARKETING_SPEND_WEEKLY` | 1,248 | this bundle (synthetic inputs) | model, EDA, R readout |
| `SALES_WEEKLY` (row access policy) | 312 | this bundle | model, EDA, R readout |
| `CONTROLS_WEEKLY` | 156 | this bundle | model, EDA, R cross-check |
| `SYNTHETIC_TRUE_PARAMETERS` | 8 | this bundle (the generator's answer key) | model, R cross-check |
| `MMM_POSTERIOR_DRAWS` | 3,200 | `ml/train_model.py` | app (scenario engine) |
| `MMM_RESPONSE_CURVES` | 640 | `ml/train_model.py` | app, EDA |
| `MMM_CONTRIBUTIONS_WEEKLY` | 1,404 | `ml/train_model.py` | app, R readout |
| `MMM_CHANNEL_SUMMARY` | 8 | `ml/train_model.py` | app, EDA, both R reports |
| `MMM_FIT` | 156 | `ml/train_model.py` | EDA |
| `MMM_PARAMETER_RECOVERY` | 24 | `ml/train_model.py` | EDA |
| `MMM_MODEL_RUN` | 1 | `ml/train_model.py` | app, EDA (model version, fit quality) |
| `MMM_R_CROSSCHECK` | 8 | `r/model_crosscheck.qmd` | R readout |
| `MMM_SAVED_SCENARIOS` | 0 | the app ("Save scenario") | app (Compare scenarios), R readout |

The bundle includes the model outputs from the current fit (v3.2), so the app shows
the exact numbers in `posit-README.md` right after loading. No refit is needed.

## Load it: pick one

**A. Snowsight only (no tools).** Open a new SQL worksheet, choose a warehouse, paste
[`lumenvale_mmm_setup.sql`](lumenvale_mmm_setup.sql) (about 500 KB) and click
**Run All**. It creates the database, the tables and the row access policy, inserts all
the data, and ends with a row count per table.

**B. From Posit Workbench (Python).** This uses Workbench-managed Snowflake credentials:

```bash
uv run python snowflake_setup/load_to_snowflake.py
```

**C. Snowflake CLI.** This uses `PUT` + `COPY INTO` from the CSVs:

```bash
snow sql -f snowflake_setup/schema.sql
snow sql -f snowflake_setup/load_from_stage.sql
```

**D. Snowsight "Load Data" dialog.** Run `schema.sql`, then load each file in
[`data/`](data/) into the table with the same name. The files have a header row,
comma separators, ISO dates, TRUE/FALSE booleans, and empty fields for NULL.

## Then point the code at it

No settings are needed. `MMM_DATA_SOURCE` defaults to `auto`, so every component uses
Snowflake when credentials are available and local files otherwise. Set
`MMM_DATA_SOURCE=snowflake` to fail instead of falling back. The app header, the EDA
footer and the R reports always say which source they used.

| Where | Credentials |
|---|---|
| Posit Workbench | Managed Snowflake credentials (Python `connection_name="workbench"`, R `odbc::snowflake()`) |
| Posit Connect: Streamlit app | Viewer's own Snowflake identity (OAuth integration, `PositAuthenticator`) |
| Posit Connect: Quarto EDA and R reports | Publisher's Snowflake integration, or the local-file fallback |

Optional environment variables (same names in Python and R): `MMM_DATABASE`,
`MMM_SCHEMA`, `SNOWFLAKE_WAREHOUSE` and `SNOWFLAKE_ACCOUNT`.

## Show the pipeline live instead

Load only the inputs, then fit the model in Snowflake:

```bash
uv run python snowflake_setup/load_to_snowflake.py --inputs-only
uv run python ml/train_model.py                  # reads inputs, writes the 7 MMM_* tables
cd r && quarto render model_crosscheck.qmd       # writes MMM_R_CROSSCHECK
```

Until the model has been fitted, the app reads the empty output tables from the
local files and its header says so. A refit gives very slightly different numbers
from those in the demo script (MCMC), so do it well before the webinar, not during.

## Access and the row access policy

- `schema.sql` runs `CREATE DATABASE IF NOT EXISTS LUMENVALE_MMM`, so load with a role
  that has `CREATE DATABASE` on the account (for example `SYSADMIN`), or ask an admin to
  create the database first and grant your role ownership of it. To use a different
  database or schema, set `MMM_DATABASE` / `MMM_SCHEMA` and edit the first lines of
  `schema.sql`.
- `SALES_CHANNEL_POLICY` shows `demo_snowflake_user@posit.co` only the **Online** rows
  of `SALES_WEEKLY`. That user sees lower revenue, and a refit run as that user models
  online sales only. Load and refit with an unrestricted user.
- Viewers need `SELECT` on the schema and `INSERT` on `MMM_SAVED_SCENARIOS` to save
  scenarios. Whoever refits needs `INSERT` and `DELETE` on the other tables. Example
  `GRANT`s are at the end of `schema.sql`.
- Re-running `schema.sql` recreates every table except `MMM_SAVED_SCENARIOS`. To clear
  scenarios: `DELETE FROM LUMENVALE_MMM.PUBLIC.MMM_SAVED_SCENARIOS;`

## After you change the data or refit locally

```bash
uv run python snowflake_setup/build_bundle.py    # rebuilds data/*.csv and the two .sql files
```

The builder checks every table's columns against `schema.sql` and fails on a mismatch.
