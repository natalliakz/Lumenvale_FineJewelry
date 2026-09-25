# Channel Investment Planner: Lumenvale Jewelers

> **This project contains synthetic data and analysis created for demonstration
> purposes only.** Lumenvale Jewelers is a fictional omnichannel fine-jewelry
> retailer.

A marketing mix model (MMM) estimates how much revenue each marketing channel drives,
how long its effect lasts, and where the next dollar would do the most good. Most MMMs
end up as a static slide deck, rebuilt from one analyst's laptop every quarter. This
project shows the alternative:

- The model code lives in Git and reads its data from **Snowflake**.
- A Bayesian MMM is fit once, and its results (response curves, contributions, ROI and
  marginal ROI, with uncertainty) are written back to Snowflake.
- A **Streamlit** app on **Posit Connect** lets the growth team plan a budget, apply
  their own business constraints and compare scenarios in real time.
- **Quarto** reports replace the old static PDF, and an **R** weekly readout is
  emailed to stakeholders on a schedule.

## What's inside

| Component | File(s) | What it does |
|---|---|---|
| Synthetic data | `data/generate_data.py` | Three years of weekly spend, sales and market controls for 8 channels. The true parameters are saved to JSON so the model can be graded. |
| Snowflake setup | `snowflake_setup/` | Schema, row access policy, and every table's data as CSV plus ready-to-run SQL ([README](snowflake_setup/README.md)) |
| Data access | `mmm_data.py`, `r/R/lumenvale.R` | One reader/writer for Snowflake or local files, used by every Python and R component |
| Bayesian MMM | `ml/model_utils.py`, `ml/train_model.py` | PyMC model with adstock, saturation, seasonality and controls. Back-tested on the last 26 weeks. Writes 6 output tables. |
| Model Registry | `ml/register_model.py`, `ml/mmm_custom_model.py`, `mmm_registry.py` | Logs the fitted model to the Snowflake Model Registry as `LUMENVALE_MMM` (version `V3_2`); the app runs its default version in a warehouse |
| Scenario engine | `planner.py` | Projects revenue for any budget from 400 posterior draws and optimises within constraints (SciPy) |
| App | `app.py`, `brand.py`, `.streamlit/config.toml` | **Channel Investment Planner** (Streamlit) |
| EDA report | `eda.qmd` | Short visual report: seasonality, carryover, fit, recovered vs true parameters, response curves |
| R weekly readout | `r/weekly_readout.qmd` | Quarto email report: last week's KPIs, where the next dollar works, and the team's saved scenarios |
| R model cross-check | `r/model_crosscheck.qmd` | An independent regression in R that checks the Python model's budget calls |
| Model outputs | `outputs/` | CSV copies of the model output tables, so the app runs without a refit |

## Getting started

### Python (data, model, app, EDA)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync                                   # install dependencies from uv.lock
uv run python data/generate_data.py       # synthetic data (runs automatically if missing)
uv run python ml/train_model.py           # fit the MMM (about 1 minute), writes outputs/ (and Snowflake)
uv run ml/register_model.py               # log it to the Snowflake Model Registry (needs Snowflake)
uv run streamlit run app.py               # launch the planner
uv run quarto render eda.qmd              # render the EDA report
```

The model outputs are already in `outputs/`, so you can go straight to the app.

### R (weekly readout and model cross-check)

Requires R 4.6 (built and tested on R 4.4) and Quarto.

```bash
cd r
Rscript -e 'renv::restore()'
quarto render model_crosscheck.qmd
quarto render weekly_readout.qmd          # also writes email-preview/index.html
```

### Snowflake

All data lives in `LUMENVALE_MMM.PUBLIC`. Load the schema and every table once,
with any of the options in [`snowflake_setup/README.md`](snowflake_setup/README.md):

```bash
# Snowsight only: paste snowflake_setup/lumenvale_mmm_setup.sql into a worksheet, Run All
# or, from Posit Workbench:
uv run python snowflake_setup/load_to_snowflake.py
```

After that, no settings are needed. Every component (model, app, EDA and R reports)
reads and writes Snowflake when credentials are available, and falls back to the
local files otherwise. `MMM_DATA_SOURCE` controls this: `auto` (default), `snowflake`
(no fallback) or `local`. On Posit Workbench, managed Snowflake credentials are picked
up automatically. On Posit Connect, the app uses each viewer's own Snowflake
credentials (OAuth integration).

## Snowflake Model Registry

The fitted model is also a Snowflake object: `LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM`, with
one version per fit (`V3_2` for model v3.2) and the fit quality from `MMM_MODEL_RUN` as
its metrics. It is a snowflake-ml `CustomModel` (`ml/mmm_custom_model.py`) that carries
the 400 posterior draws and the scenario code in `planner.py`, and it has two methods:

| Method | Input (one row) | Output |
|---|---|---|
| `RECOMMEND` | `EXTRA_BUDGET`, then a cap per channel (`CAP_PAID_SEARCH`, …; `NULL` = no cap) | `ADD_<channel>` per channel, `INCREMENTAL` (with `_P05`/`_P95`), `ROAS`, `NOTE` |
| `PREDICT` | quarterly $ per channel (`PAID_SEARCH`, `CONNECTED_TV`, …) | `SPEND`, `REVENUE` and `INCREMENTAL` (each with `_P05`/`_P95`), `ROAS` |

```bash
uv run ml/train_model.py                  # refit; writes the MMM_* tables
uv run ml/register_model.py               # log that fit as a new version and make it the default
```

`register_model.py` runs on **Python 3.12** in its own uv environment (the script
header lists its dependencies). The registry stores the model class as bytecode, so
it must be logged on the Python the warehouse runs it on. It ends by calling
`RECOMMEND` in Snowflake and checking the answer against the local calculation.

Any Snowflake user can call the model from SQL, so the same numbers are available
outside the app:

```sql
SHOW VERSIONS IN MODEL LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM;
-- +$200K with Connected TV capped at +$60K (arguments in channel order: Paid Search,
-- Connected TV, Paid Social, Display, Affiliate, Email / CRM, Direct Mail, Podcast)
SELECT MODEL(LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM, V3_2)!RECOMMEND(
    200000, NULL, 60000, NULL, NULL, NULL, NULL, NULL, NULL);
-- Roll back: the app follows the default version within a minute
ALTER MODEL LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM SET DEFAULT_VERSION = V3_1;
```

The app finds the default version with `SHOW MODELS` / `SHOW VERSIONS` and calls it
with the SQL above, over the same Snowflake connection it uses for the tables. On
Connect that is the viewer's own identity, and the app needs no extra packages.
`MMM_SCORING` controls this: `auto` (default: the registered model if there is one,
otherwise the same code locally on `MMM_POSTERIOR_DRAWS`), `registry` (no fallback)
or `local`. Viewers need `USAGE` on the model; see the grants at the end of
`snowflake_setup/schema.sql`.

## Snowflake tables

| Table | Grain | Written by |
|---|---|---|
| `MARKETING_SPEND_WEEKLY` | week × channel: spend, impressions | data generator |
| `SALES_WEEKLY` | week × sales channel (Online / Showroom): orders, revenue, AOV | data generator |
| `CONTROLS_WEEKLY` | week: promo and holiday flags, consumer confidence, diamond price index | data generator |
| `MMM_POSTERIOR_DRAWS` | draw × channel: decay, saturation, effect size | Python model |
| `MMM_RESPONSE_CURVES` | channel × spend level: revenue with 90% band, marginal ROI | Python model |
| `MMM_CONTRIBUTIONS_WEEKLY` | week × component: revenue contribution | Python model |
| `MMM_CHANNEL_SUMMARY` | channel: spend, ROI, marginal ROI (with intervals), carryover | Python model |
| `MMM_FIT` | week: actual, predicted, back-test | Python model |
| `MMM_PARAMETER_RECOVERY` | channel × parameter: estimate vs true value | Python model |
| `MMM_MODEL_RUN` | one row: model version, fit date, fit quality (the app's model badge) | Python model |
| `SYNTHETIC_TRUE_PARAMETERS` | channel: the generator's true decay, saturation, ROI | data generator |
| `MMM_SAVED_SCENARIOS` | scenario × channel: plan and projection (reloaded into Compare scenarios) | Streamlit app |
| `MMM_R_CROSSCHECK` | channel: R vs Python ROI and verdict | R cross-check |

## The app

**Channel Investment Planner** ("Marketing mix model · what-if analysis for channel
investment · VP of Growth") has four tabs. The **model v3.2** badge in the header is
the default version of `LUMENVALE_MMM` in the Snowflake Model Registry, and the label
under it says so. Before the model is registered (or offline), the badge comes from
`MMM_MODEL_RUN`, the row `ml/train_model.py` writes on every fit.

1. **Scenario**: the VP of Growth's view. Pick an **Additional budget** ($0–$500K) for
   the horizon quarter (the next calendar quarter, e.g. Q4 2026; override with
   `MMM_HORIZON`), set caps on the extra spend per channel under **Constraints**
   (default: Connected TV capped at +$60K), and click **Run scenario**. It shows the
   recommended allocation by channel, the **projected incremental revenue** and the
   **blended ROAS** on the additional budget. **Run scenario** calls the registered
   model's `RECOMMEND` method in Snowflake, and the line under the results names the
   model and version that answered. **Save to compare** keeps it.
2. **Where the money goes today**: spend, media-driven revenue, contribution over
   time, and ROI vs marginal ROI by channel.
3. **Plan the full mix**: set a quarterly budget, adjust channels with sliders, set
   minimum and maximum spend per channel in **Business constraints**, then click
   **Optimize within my constraints**. Projected revenue and ROAS are shown with 90%
   intervals, next to each channel's response curve.
4. **Compare scenarios**: save named scenarios and compare them side by side.

Every interaction is computed from pre-calculated posterior draws, so the app never
refits the model. The other tabs run locally on `MMM_POSTERIOR_DRAWS`, so the sliders
respond instantly. The app warns if those draws are from a different version than the
registry's default.

## Custom branding with `_brand.yml`

Colours and fonts are defined once in `_brand.yml` ([brand.yml](https://posit-dev.github.io/brand-yml/)):
a Posit + Snowflake co-branded theme (Posit navy, blue and orange with Snowflake blue;
Open Sans and Lato). The 8 channel colours in `brand.py` (and `r/R/lumenvale.R`) come
from the same brand hues, adjusted to pass colour-blind separation checks, and each
channel keeps its colour on every chart.

- **Quarto** (`eda.qmd`, and the R reports via `r/_brand.yml`) applies it automatically.
- **Streamlit** doesn't read `_brand.yml` directly, so `brand.py` turns it into the app
  theme. After editing `_brand.yml`, run `uv run python brand.py` to regenerate
  `.streamlit/config.toml`. Then run `r/sync_snapshot.sh` to copy the file into `r/`.

## Use it as inspiration

- Replace the generator with your own spend and sales tables: the model only needs
  weekly spend by channel, weekly revenue, and a few controls.
- Swap the scenario rules for your own, such as minimum brand spend, agency
  commitments or channel caps. The optimiser takes simple min/max bounds.
- Schedule `ml/train_model.py` on Posit Connect to refit monthly, and let the app and
  the R readout pick up the new outputs automatically.

## Important Disclaimer

**This project contains synthetic data and analysis created for demonstration
purposes only.**

All data, insights, business scenarios, and analytics presented in this
demonstration project have been artificially generated using AI. The data does
not represent actual business information, performance metrics, customer data,
or operational statistics.

### Key Points:

- **Synthetic Data**: All datasets are computer-generated and designed to
  illustrate analytical capabilities
- **Illustrative Analysis**: Insights and recommendations are examples of the
  types of analysis possible with Posit tools
- **No Actual Business Data**: No real business information or data was used or
  accessed in creating this demonstration
- **Educational Purpose**: This project serves as a technical demonstration of
  data science workflows and reporting capabilities
- **AI-Generated Content**: Analysis, commentary, and business scenarios were
  created by AI for illustration purposes
- **No Real-World Implications**: The scenarios and insights presented should
  not be interpreted as actual business advice or strategies

This demonstration showcases how Posit's commercial and open-source tools can be
applied to the fine jewelry retail industry. The synthetic data and analysis provide a
foundation for understanding the potential value of implementing similar
analytical workflows with actual business data.

For questions about adapting these techniques to your real business scenarios,
please contact your Posit representative.
---

*This demonstration was created using Posit's commercial data science tools and
open-source packages. All synthetic data and analysis are provided for
evaluation purposes only.*
