-- Lumenvale Jewelers (FICTIONAL) - Channel Investment Planner
-- Snowflake schema and tables for the demo.
--
-- Run it on its own (Snowsight worksheet: "Run All"), or use one of the loaders in
-- snowflake_setup/README.md, which run this file first and then load the data.
-- Re-running it recreates every table EXCEPT MMM_SAVED_SCENARIOS, so scenarios
-- saved from the app survive a reload.
--
-- DISCLAIMER: This project contains synthetic data and analysis created for
-- demonstration purposes only.

CREATE DATABASE IF NOT EXISTS LUMENVALE_MMM;  -- tables go in its default PUBLIC schema
USE SCHEMA LUMENVALE_MMM.PUBLIC;

-- ---------------------------------------------------------------------------
-- Model inputs: what marketing ops and finance load every week
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE MARKETING_SPEND_WEEKLY (
    WEEK_START DATE,                -- Monday of the week
    CHANNEL VARCHAR,                -- one of 8 media channels
    SPEND NUMBER(14, 2),            -- USD
    IMPRESSIONS NUMBER(14, 0)
);

CREATE OR REPLACE TABLE SALES_WEEKLY (
    WEEK_START DATE,
    SALES_CHANNEL VARCHAR,          -- Online / Showroom
    ORDERS NUMBER(10, 0),
    REVENUE NUMBER(14, 2),          -- USD
    AVG_ORDER_VALUE NUMBER(10, 2)
);

CREATE OR REPLACE TABLE CONTROLS_WEEKLY (
    WEEK_START DATE,
    PROMO_FLAG NUMBER(1, 0),        -- 1 = sitewide promotion that week
    HOLIDAY_FLAG NUMBER(1, 0),      -- 1 = gifting holiday in the week
    CONSUMER_CONFIDENCE_INDEX FLOAT,
    DIAMOND_PRICE_INDEX FLOAT
);

-- The generator's answer key. Only exists because the data is synthetic; the
-- model is graded against it (MMM_PARAMETER_RECOVERY).
CREATE OR REPLACE TABLE SYNTHETIC_TRUE_PARAMETERS (
    CHANNEL VARCHAR, DECAY FLOAT, K FLOAT, BETA FLOAT, TOTAL_SPEND FLOAT,
    TOTAL_CONTRIBUTION FLOAT, ROI FLOAT, MARGINAL_ROI_AT_RUN_RATE FLOAT
);

-- ---------------------------------------------------------------------------
-- Model outputs: written by ml/train_model.py (reloaded on every fit)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE MMM_POSTERIOR_DRAWS (
    DRAW NUMBER, CHANNEL VARCHAR, DECAY FLOAT, K FLOAT, BETA FLOAT
);

CREATE OR REPLACE TABLE MMM_RESPONSE_CURVES (
    CHANNEL VARCHAR, WEEKLY_SPEND FLOAT, REVENUE FLOAT, REVENUE_P05 FLOAT,
    REVENUE_P95 FLOAT, MARGINAL_ROI FLOAT, WITHIN_OBSERVED_RANGE BOOLEAN
);

CREATE OR REPLACE TABLE MMM_CONTRIBUTIONS_WEEKLY (
    WEEK_START DATE, COMPONENT VARCHAR, CONTRIBUTION FLOAT
);

CREATE OR REPLACE TABLE MMM_CHANNEL_SUMMARY (
    CHANNEL VARCHAR, TOTAL_SPEND FLOAT, CONTRIBUTION FLOAT, CONTRIBUTION_P05 FLOAT,
    CONTRIBUTION_P95 FLOAT, ROI FLOAT, ROI_P05 FLOAT, ROI_P95 FLOAT,
    RUN_RATE_WEEKLY_SPEND FLOAT, CURRENT_QUARTER_BUDGET FLOAT,
    MAX_OBSERVED_WEEKLY_SPEND FLOAT, MARGINAL_ROI FLOAT, MARGINAL_ROI_P05 FLOAT,
    MARGINAL_ROI_P95 FLOAT, DECAY FLOAT
);

CREATE OR REPLACE TABLE MMM_FIT (
    WEEK_START DATE, ACTUAL FLOAT, PREDICTED FLOAT, PREDICTED_P05 FLOAT,
    PREDICTED_P95 FLOAT, IN_HOLDOUT_BACKTEST BOOLEAN, BACKTEST_PREDICTED FLOAT
);

CREATE OR REPLACE TABLE MMM_PARAMETER_RECOVERY (
    CHANNEL VARCHAR, PARAMETER VARCHAR, TRUE_VALUE FLOAT, ESTIMATE FLOAT,
    ESTIMATE_P05 FLOAT, ESTIMATE_P95 FLOAT
);

-- One row: the model version and fit quality shown in the app, EDA and readout.
CREATE OR REPLACE TABLE MMM_MODEL_RUN (
    MODEL_VERSION VARCHAR, FITTED_AT TIMESTAMP_NTZ, DATA_SOURCE VARCHAR, WEEKS NUMBER,
    FIRST_WEEK DATE, LAST_WEEK DATE, R2 FLOAT, MAPE FLOAT, HOLDOUT_WEEKS NUMBER,
    HOLDOUT_MAPE FLOAT, MAX_RHAT FLOAT, DRAWS NUMBER, RUN_RATE_WEEKS NUMBER,
    QUARTER_WEEKS NUMBER
);

-- Written by r/model_crosscheck.qmd: the R model's second opinion per channel.
CREATE OR REPLACE TABLE MMM_R_CROSSCHECK (
    CHANNEL VARCHAR, R_DECAY FLOAT, R_ROI FLOAT, R_MROI FLOAT, PY_ROI FLOAT,
    PY_MROI FLOAT, R_VERDICT VARCHAR, PY_VERDICT VARCHAR, AGREE BOOLEAN,
    CHECKED_AT TIMESTAMP_NTZ
);

-- Scenarios saved from the Streamlit app (one row per scenario x channel); the
-- weekly R readout reports on them. Not recreated, so saved scenarios survive.
CREATE TABLE IF NOT EXISTS MMM_SAVED_SCENARIOS (
    SCENARIO_NAME VARCHAR, SAVED_AT TIMESTAMP_NTZ, SAVED_BY VARCHAR, CHANNEL VARCHAR,
    QUARTERLY_BUDGET FLOAT, PROJECTED_REVENUE FLOAT, TOTAL_BUDGET FLOAT,
    TOTAL_REVENUE FLOAT, TOTAL_REVENUE_P05 FLOAT, TOTAL_REVENUE_P95 FLOAT,
    BLENDED_ROI FLOAT, CHANGE_VS_CURRENT FLOAT
);

-- ---------------------------------------------------------------------------
-- Row access policy
-- The shared demo user may see ONLINE sales only; everyone else sees all rows.
-- Showroom revenue is hidden from demo_snowflake_user@posit.co, so its totals
-- in SALES_WEEKLY are lower than the local CSVs. See posit-README.md.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE ROW ACCESS POLICY SALES_CHANNEL_POLICY
    AS (SALES_CHANNEL VARCHAR) RETURNS BOOLEAN ->
    CASE
        WHEN UPPER(CURRENT_USER()) = 'DEMO_SNOWFLAKE_USER@POSIT.CO' THEN SALES_CHANNEL = 'Online'
        ELSE TRUE
    END;

ALTER TABLE SALES_WEEKLY ADD ROW ACCESS POLICY SALES_CHANNEL_POLICY ON (SALES_CHANNEL);

-- ---------------------------------------------------------------------------
-- Optional: access for the role your Workbench / Connect users run as.
-- Replace DEMO_ROLE and uncomment. Viewers need INSERT on MMM_SAVED_SCENARIOS
-- to save scenarios; whoever refits the model needs INSERT/DELETE on the rest.
-- ---------------------------------------------------------------------------
-- GRANT USAGE ON DATABASE LUMENVALE_MMM TO ROLE DEMO_ROLE;
-- GRANT USAGE ON SCHEMA LUMENVALE_MMM.PUBLIC TO ROLE DEMO_ROLE;
-- GRANT SELECT ON ALL TABLES IN SCHEMA LUMENVALE_MMM.PUBLIC TO ROLE DEMO_ROLE;
-- GRANT INSERT ON TABLE LUMENVALE_MMM.PUBLIC.MMM_SAVED_SCENARIOS TO ROLE DEMO_ROLE;
-- GRANT INSERT, DELETE ON ALL TABLES IN SCHEMA LUMENVALE_MMM.PUBLIC TO ROLE DEMO_ROLE;
--
-- Snowflake Model Registry (ml/register_model.py logs LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM):
-- whoever registers the model needs CREATE MODEL on the schema; app viewers need USAGE
-- on the model (and on a warehouse) to run it from the Streamlit app.
-- GRANT CREATE MODEL ON SCHEMA LUMENVALE_MMM.PUBLIC TO ROLE DEMO_ROLE;
-- GRANT USAGE ON MODEL LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM TO ROLE DEMO_ROLE;  -- after registering
-- GRANT USAGE ON WAREHOUSE DEFAULT_WH TO ROLE DEMO_ROLE;
