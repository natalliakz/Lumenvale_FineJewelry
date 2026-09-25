# Posit Demo Guide: Channel Investment Planner (Lumenvale Jewelers)

> **This project contains synthetic data and analysis created for demonstration
> purposes only.** Lumenvale Jewelers is fictional. Do not name the prospect, its
> products or its vendors on screen, and do not present any number here as theirs.

**Format:** 5-minute Posit + Snowflake webinar segment
**Audience:** marketing analytics, growth and data leaders at an omnichannel retailer
**Posit products:** Posit Workbench, Posit Connect, Quarto, Streamlit (Python), R
**Partner:** Snowflake (inputs and model outputs live in `LUMENVALE_MMM.PUBLIC`)
**Look:** Posit + Snowflake co-branded theme (`_brand.yml`), used by the app and every report

---

## 1. The story in one breath

**Before:** the marketing mix model ran on one analyst's laptop. Once a quarter it
became a static 20-page report. When the VP of Growth asked "what if we move $500K
from Search to CTV?", the answer took two weeks.

**After:** the code is in Git and reads from Snowflake. The model is fit once and its
results, and its version, are written back to Snowflake. The VP of Growth opens the
**Channel Investment Planner** on Posit Connect, asks "where should the next $200K go?",
adds what they know that the model doesn't, and gets an answer in under a second.

> *"The data scientist owns the model; the business owns the decision."*

## 2. Industry context

Fine-jewelry retail is a high-consideration, highly seasonal purchase. December is a
large share of the year, and there is a smaller peak around Valentine's Day and
Mother's Day. Customers research for weeks, so awareness channels (Connected TV,
Podcast) pay back slowly. Direct-response channels (Paid Search, Email / CRM) convert
within days. The same seasonality makes naive spend-vs-revenue charts misleading:
budget rises when demand does anyway. That is why the model is needed.

## 3. The data (synthetic)

- **156 weeks**, from 2023-01-02 to 2025-12-22, for **8 channels**: Paid Search,
  Connected TV, Paid Social, Display, Affiliate, Email / CRM, Direct Mail, Podcast.
- **Sales** are split into Online and Showroom. **Controls** include promo weeks,
  holidays, consumer confidence and a diamond price index.
- **Built-in truths:**
  - CTV and Podcast have long carryover (adstock); Search and Email / CRM are near-immediate.
  - **Paid Search is saturated** (heavy spend on the flat part of its curve).
  - **Connected TV is under-invested** (steep part of its curve).
- The true parameters are in `data/synthetic-true_parameters.json`. The EDA shows the
  model recovering them, which is a strong answer to "how do you know the model is right?"

**Model quality (v3.2):** R² 94.7%, MAPE 1.9%. On a 26-week back-test that the model
never saw, MAPE is 1.9%. Max R-hat is 1.003 and there are 400 posterior draws.

## 4. Pre-demo checklist

**Once, a day or more before:** load the schema and data into Snowflake with any
option in [`snowflake_setup/README.md`](snowflake_setup/README.md). The simplest is to
paste `snowflake_setup/lumenvale_mmm_setup.sql` into a Snowsight worksheet and click
**Run All**. The bundle includes the model outputs behind every number below.

Run through this list 30 minutes before the webinar.

- [ ] The app on Connect opens. Top right shows the **model v3.2** badge with
      **Snowflake · LUMENVALE_MMM.PUBLIC.MMM_MODEL_RUN** under it, and
      **Data: Snowflake · LUMENVALE_MMM.PUBLIC** (not "local file" / "Local files").
- [ ] Reload the page so the **Scenario** tab is on its defaults: **+$200K**, Connected TV
      capped at +$60K.
- [ ] **Plan the full mix** is on the **current plan** (click **Reset to current plan** and
      **Reset constraints**).
- [ ] **Compare scenarios** has no leftover scenarios from rehearsal. Saved scenarios are loaded back
      from Snowflake, so restarting the app does not clear them. Run: `DELETE FROM
      LUMENVALE_MMM.PUBLIC.MMM_SAVED_SCENARIOS;`
- [ ] The Snowflake worksheet is open in a second tab, with these queries ready:
      ```sql
      SELECT * FROM LUMENVALE_MMM.PUBLIC.MARKETING_SPEND_WEEKLY ORDER BY WEEK_START DESC LIMIT 20;
      SELECT * FROM LUMENVALE_MMM.PUBLIC.MMM_CHANNEL_SUMMARY;
      SELECT MODEL_VERSION, FITTED_AT, WEEKS, HOLDOUT_MAPE FROM LUMENVALE_MMM.PUBLIC.MMM_MODEL_RUN;
      ```
- [ ] The Git repository is open in a third tab (for the "it's in Git" moment).
- [ ] The EDA report ("what used to be the 20-page report") is published and open.
- [ ] Browser zoom is 100–110%, notifications are off, and the window is 1440 px wide or more.
- [ ] Do one dry run of the Scenario and Plan-the-full-mix moves below (about 20 seconds).

## 5. The 5-minute demo script

The numbers below are for the current model (**v3.2**), which is exactly what the
Snowflake bundle loads. If you run as `demo_snowflake_user@posit.co`, see
[the row access limitation](#7-snowflake-notes-and-limitations).

### 0:00 – The before (30 s)

> **"This model used to live on one laptop and ship as a static report."**

- Briefly show the EDA report and say: "This is what used to be the 20-page report.
  It was out of date on the day it shipped, and nobody could ask it a question."
- "Today the code is in Git, the data is in Snowflake, and the answers are in an app."

### 0:30 – Snowflake is the source of truth (40 s)

- Open the Snowflake worksheet. Show `MARKETING_SPEND_WEEKLY`: weekly spend by
  channel, as marketing ops loads it.
- Show `MMM_CHANNEL_SUMMARY`: "The model writes its results back here: ROI, marginal
  ROI and uncertainty for every channel, plus response curves and posterior draws."
- Run the `MMM_MODEL_RUN` query: "And this is the model itself: **v3.2**, when it was
  fit, and its hold-out error. Remember that version number."
- *Talking point:* the model is fit on Workbench with managed Snowflake credentials.
  There are no passwords in the code.

### 1:10 – "Scenario": where should the next $200K go? (55 s)

- Open the **Channel Investment Planner** on Connect. Point at the top right: "**model
  v3.2**, read from `MMM_MODEL_RUN` in Snowflake. It's the same row we just queried, so
  the VP of Growth always knows which model they're looking at. Refit the model and the
  badge changes on the next page load, with no redeploy."
- The **Scenario** panel: **Additional budget +$200K**, **Horizon Q4 2026**, and one
  business constraint: **Connected TV capped at +$60K** (inventory is nearly sold out).
  Click **Run scenario**.
- **Recommended allocation of +$200K:** Email / CRM **+$115K**, Connected TV **+$60K**
  (at its cap), Affiliate **+$13K**, Display **+$12K**. **Paid Search gets $0.**
- **Projected incremental revenue $567K** (90% interval $286K–$983K), a **blended ROAS
  of 2.8×** on the additional $200K.
- Optional: drag to **$500K** and **Run scenario**: **$1.14M** at **2.3×**. "Every
  extra dollar returns a little less. That's saturation, and the model shows it."
- Click **Save to compare**.

### 2:05 – "Where the money goes today": why not Paid Search? (45 s)

- Top line: **$25.05M** of media spend over the last 52 weeks drove **$74.80M** of
  revenue, a **3.0×** blended ROI. Media accounts for **42%** of revenue.
- In **Return on spend by channel**: "Paid Search looks like a hero, with a **4.1×**
  average ROI. But its *marginal* ROI, what the next dollar returns, is **0.82×**. It's
  saturated. Connected TV's next dollar returns **3.7×**, the best of any channel."
- *Talking point:* average ROI tells you what worked. Marginal ROI tells you where the
  next dollar should go. A static report usually only shows the first.

### 2:50 – "Plan the full mix": move money Search → CTV (40 s)

- The current plan is **$6.26M per quarter**, projecting **$18.94M** of media-driven
  revenue (90% interval $14.16M–$24.35M) at **3.03×** ROAS.
- Move **Paid Search** down by $500K (2,160 → 1,660) and **Connected TV** up by $500K
  (370 → 870). "Same budget: **$19.99M**, **+$1.04M (+5.5%)**, at **3.19×** ROAS."
- Name it **"Shift $500K Search to CTV"** → **Save scenario**.

### 3:30 – Business constraints + Optimize (45 s)

- **Reset to current plan**. Click **✦ Optimize within my constraints** with the
  default guardrails (each channel between 50% and 3× its current budget): **$20.78M
  (+$1.83M)**, and the optimizer pushes CTV to its cap.
- "But the VP of Growth knows CTV inventory sells out past $900K a quarter." Set
  **Connected TV Max ($K)** to **900** and click **Optimize** again: **$20.56M, +$1.62M
  (+8.6%)** at **3.28×** ROAS. Save it as **"Optimized within guardrails"**.
- "The optimizer gave up about $200K of theoretical upside to respect a real-world
  limit. The business knowledge wins."

### 4:15 – "Compare scenarios" (30 s)

- Current plan vs "Recommended +$200K (Q4 2026)" vs "Shift $500K Search to CTV" vs
  "Optimized within guardrails", side by side with 90% intervals.
- "Saved scenarios are written to Snowflake, so finance sees the same numbers. The R
  weekly readout emails them to leadership every Monday."

### 4:45 – Close (15 s)

> **"The data scientist owns the model; the business owns the decision."**

- The model is in Git, its outputs and version are in Snowflake, and the app is on
  Posit Connect, secured with each viewer's own Snowflake identity.

### If you have more time: R bonus material

- **`r/model_crosscheck.qmd`** is a second opinion: an independent R regression agrees
  with the Bayesian model on 6 of 8 budget calls (Spearman 0.81 on marginal ROI), and
  both say CTV beats Search. It shows Python and R teams working from the same
  Snowflake tables.
- **`r/weekly_readout.qmd`** is a Quarto email report, scheduled on Connect. It covers
  last week's KPIs, where the next dollar works, and saved scenarios, and arrives in
  the inbox with the chart inline.

## 6. Key insights to land

1. **The next $200K doesn't go to Paid Search:** the model sends it to Email / CRM and
   Connected TV, within the business's CTV cap.
2. **Paid Search is saturated:** a 4.1× average ROI but only a 0.82× marginal ROI.
3. **CTV is under-invested:** a 3.7× marginal ROI, but a long carryover means
   last-click reporting misses most of its value.
4. **Reallocating is free money:** at the same budget, moving $500K adds about $1.0M
   of quarterly revenue.
5. **The constraints matter:** the optimizer respects what the business knows, and
   the trade-off it makes is visible.
6. **Uncertainty is shown, not hidden:** every projection has a 90% interval.
7. **One model version everywhere:** the badge in the app, the EDA footer and
   `MMM_MODEL_RUN` in Snowflake all say v3.2.

**Be ready for the question "is the model perfect?" No, and that's the point:**

- The Python model over-credits **Display** (true marginal ROI is about 0.9×; the
  model says 2.1×) and **Affiliate**. If asked, say: "This is exactly why the business
  constraints exist: put a cap on Display." The parameter recovery chart in the EDA
  shows this openly.
- The R cross-check disagrees on Affiliate (Python: grow; R: hold) and Podcast (Python:
  hold; R: grow). Its **Email / CRM** estimate is unreliable: that spend is small and moves
  with December, so R's 9.2× marginal ROI is shown as off-scale and flagged in the report.

## 7. Snowflake notes and limitations

- Everything is in **`LUMENVALE_MMM.PUBLIC`**: 13 tables, which are the inputs,
  the answer key, 7 model-output tables, the R cross-check and saved scenarios.
  [`snowflake_setup/`](snowflake_setup/README.md) has the schema (`schema.sql`), the
  data as CSVs, and three ways to load it: a Snowsight-only all-in-one SQL file, a
  Workbench Python loader, and `PUT`/`COPY` for the Snowflake CLI.
- **Every component reads and writes these tables.** The model reads the inputs and
  writes its outputs and `MMM_MODEL_RUN`. The app reads the outputs and writes and
  re-reads `MMM_SAVED_SCENARIOS`. The EDA reads everything. The R cross-check writes
  `MMM_R_CROSSCHECK`, and the R readout reads it along with the saved scenarios.
- **Row access policy:** `SALES_CHANNEL_POLICY` on `SALES_WEEKLY` shows
  `demo_snowflake_user@posit.co` only the **Online** rows. When you connect as that
  user, revenue totals are lower than the numbers in this guide, and a model refit from
  Snowflake gives different estimates. This is a nice security talking point ("same
  app, each viewer sees only what they're allowed to"). For the numbers above, either
  present with an unrestricted user, or keep the model outputs from the local fit (the
  output tables are not covered by the policy).
- **Credentials:**
  - **Workbench:** managed credentials (`connection_name="workbench"` in Python,
    `odbc::snowflake()` in R).
  - **Connect (Python app):** viewer-level OAuth through the Posit SDK
    `PositAuthenticator`. Add the Snowflake OAuth integration to the content item.
  - **Connect (R reports):** a service-account integration, or run offline from a
    local snapshot (see below).
- `MMM_DATA_SOURCE` can be `auto`, `snowflake` or `local`, with the same behaviour in
  Python and R:
  - `auto` (the default) uses Snowflake when credentials are available, and local files
    otherwise or for a table not loaded yet.
  - `snowflake` fails instead of falling back.

  The app header, the EDA footer and the R reports show which source is live. "(N
  table(s) from local files)" means part of the schema is not loaded.
- **Live-pipeline variant:** load only the inputs
  (`load_to_snowflake.py --inputs-only`) and run `ml/train_model.py` on Workbench to
  fill the output tables. A refit shifts the numbers slightly (MCMC), so the script's
  numbers are only guaranteed with the bundled outputs.

## 8. Setup, regenerating the data, and refitting

```bash
cd Lumenvale_FineJewelry
uv sync                                                   # Python 3.14 environment

uv run python data/generate_data.py                       # 1. regenerate synthetic data (seeded)
MMM_DATA_SOURCE=local uv run python ml/train_model.py     # 2. refit the MMM (~1 min), writes outputs/
uv run quarto render eda.qmd                              # 3. re-render the EDA
uv run streamlit run app.py                               # 4. check the app locally
uv run python snowflake_setup/build_bundle.py             # 5. rebuild the Snowflake bundle

# Snowflake (from Workbench)
uv run python snowflake_setup/load_to_snowflake.py        # schema, policy and all tables
# or: --inputs-only, then `uv run python ml/train_model.py` to fit in Snowflake

# R reports (from r/)
cd r
Rscript -e 'renv::restore()'
quarto render model_crosscheck.qmd                        # writes MMM_R_CROSSCHECK
quarto render weekly_readout.qmd                          # email preview in email-preview/index.html
```

The generator is seeded. A refit reproduces the story (Search saturated, CTV
under-invested), but individual numbers can shift by a few percent (MCMC). After a
refit, rebuild the bundle (step 5), reload Snowflake, and update the numbers in
section 5 by running the moves in the app.

## 9. Deploying to Posit Connect

**Streamlit app** (uses `requirements.txt`; PyMC isn't needed at runtime):

```bash
uv run rsconnect deploy streamlit . --entrypoint app.py --title "Channel Investment Planner" \
  --exclude "r/*" --exclude "data/*.csv" --exclude "snowflake_setup/*" --exclude ".venv/*"
```

Then, in the content settings:

- set `MMM_DATA_SOURCE=snowflake`
- add the Snowflake OAuth integration

The synthetic input CSVs aren't needed, because the app reads model outputs only. If
you want the app to run without Snowflake, leave `MMM_DATA_SOURCE` unset: it then
reads the `outputs/` CSVs in the bundle.

**EDA report:**

```bash
uv run rsconnect deploy quarto . --entrypoint eda.qmd --title "Lumenvale MMM: the old 20-page report"
```

**R reports** (deploy from `r/`):

```bash
cd r
./sync_snapshot.sh        # copies _brand.yml plus a data snapshot for the offline fallback
rsconnect deploy quarto . --entrypoint weekly_readout.qmd     # or use Publish in Workbench
rsconnect deploy quarto . --entrypoint model_crosscheck.qmd
```

- For the weekly readout, set a schedule (Mondays 7:00) and tick **Send email after
  update**. Connect uses the `::: {.email}` block as the message.
- For Snowflake on Connect, the server needs the Snowflake ODBC driver and a
  service-account integration. Without them, the reports fall back to `r/snapshot/`.

**Git-backed deployment:** commit the project, including `manifest.json`, which you
create with `rsconnect write-manifest streamlit .`. Then, on Connect, choose **Publish
→ Import from Git** so every push redeploys. That is the "code in Git" story.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| App header says "Local files (offline mode)" | `MMM_DATA_SOURCE` is unset, or the Snowflake connection failed. The error is shown in the app. On Connect, check the OAuth integration. |
| The badge says "local file outputs/model_metadata.json" | `MMM_MODEL_RUN` is missing or empty in Snowflake (or the app is offline). Load `snowflake_setup/` or refit, then reload. |
| The badge shows a different version than expected | The badge is whatever row is in `MMM_MODEL_RUN`. Check it with the query in the checklist. Set `MODEL_VERSION` in `ml/train_model.py` before a refit. |
| Header says "(N table(s) from local files)" | Those tables are missing or empty in Snowflake. Load `snowflake_setup/` (or run the model). |
| `... does not exist. Run snowflake_setup/schema.sql first.` | The code writes only into existing tables (plain `INSERT`s, no stages). Run `schema.sql`. |
| "Save scenario" warns "saved for this session only" | The viewer lacks `INSERT` on `MMM_SAVED_SCENARIOS`. See the `GRANT`s at the end of `schema.sql`. |
| Revenue numbers are lower than this guide | You're signed in as `demo_snowflake_user@posit.co`: the row access policy hides Showroom sales. |
| `Model output ... not found` | Run `uv run python ml/train_model.py`. |
| The Scenario tab says "Settings changed" | The slider or a cap was changed after the last run. Click **Run scenario**. |
| **Compare scenarios** shows old rehearsal scenarios | Clear `MMM_SAVED_SCENARIOS` (see the checklist), then reload the page. |
| Quarto email render fails with "invalid use of '%'" | Quarto's email filter can't take a literal `%` inside the `.email` block. Changes are written in words ("up 5 percent") there on purpose. |
| `renv::restore()` wants a different R version | The lockfile was recorded on R 4.4.2. Under R 4.6, restore works; run `renv::snapshot()` afterwards to record 4.6. |
| `uv` complains about the workspace | This project is excluded from the repository-root uv workspace. Run `uv` from inside `Lumenvale_FineJewelry/`. |

## 11. Version notes

- **Python 3.14.** Python 3.15 was requested but is still a release candidate, and
  the scientific stack has no wheels for it yet.
- **MMM:** the MMM is written directly in **PyMC** (geometric adstock, logistic
  saturation, Fourier seasonality, controls). `pymc-marketing` doesn't install on
  Python 3.14 yet (pydantic incompatibility). The model structure is the same, so you
  can swap it back in when a compatible release ships.
- **R 4.6** is the target. The R documents were built and tested on R 4.4.2, and
  `r/renv.lock` pins public CRAN packages, so restores work anywhere.

## 12. Next steps and call to action

- **Pilot on real data:** point `mmm_data.py` at the prospect's own spend and sales
  tables in Snowflake. The model only needs weekly spend by channel, revenue and a few
  controls.
- **Automate:** schedule `ml/train_model.py` on Connect for a monthly refit. The app
  and the weekly email pick up the new outputs without redeploying.
- **Govern:** use viewer-level Snowflake credentials plus row access policies, so each
  team sees only its own data.
- **Ask:** "Which decision would you most like your marketing team to make in the app
  rather than wait for a report?" Offer a 2-week proof of concept on Posit Workbench +
  Connect with their Snowflake account.

---

*This project contains synthetic data and analysis created for demonstration purposes
only. All company names, products, figures and scenarios are fictional.*
