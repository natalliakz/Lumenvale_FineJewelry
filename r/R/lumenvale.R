# Shared helpers for the R documents: data access, brand colours, ggplot theme.
#
# Data access mirrors the Python project (mmm_data.py): the same tables are read
# from LUMENVALE_MMM.PUBLIC in Snowflake when a connection is available, and
# from the project's local files otherwise (or for a table not loaded yet).
# Set MMM_DATA_SOURCE to "auto" (default), "snowflake" or "local". The schema is
# created and loaded with ../snowflake_setup/ (see its README).
#
# DISCLAIMER: This project contains synthetic data and analysis created for
# demonstration purposes only.

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# Same environment variables as mmm_data.py.
LV_DATABASE <- Sys.getenv("MMM_DATABASE", "LUMENVALE_MMM")
LV_SCHEMA <- Sys.getenv("MMM_SCHEMA", "PUBLIC")
LV_WAREHOUSE <- Sys.getenv("SNOWFLAKE_WAREHOUSE", "DEFAULT_WH")

LV_CHANNELS <- c("Paid Search", "Connected TV", "Paid Social", "Display",
                 "Affiliate", "Email / CRM", "Direct Mail", "Podcast")

# Same fixed channel colours as the Streamlit app (brand.py).
LV_CHANNEL_COLORS <- c(
  "Paid Search" = "#1F6FB5", "Connected TV" = "#EE6331", "Paid Social" = "#29B5E8",
  "Display" = "#72994E", "Affiliate" = "#7D44CF", "Email / CRM" = "#D98A00",
  "Direct Mail" = "#9A4665", "Podcast" = "#00A19B", "Baseline" = "#C8D1D8"
)

INPUT_TABLES <- c("MARKETING_SPEND_WEEKLY", "SALES_WEEKLY", "CONTROLS_WEEKLY")
TRUTH_TABLE <- "SYNTHETIC_TRUE_PARAMETERS"  # the generator's answer key

# Local copies: `snapshot/` (for a Connect bundle, see sync_snapshot.sh) first,
# then the project folder one level up.
LV_LOCAL_ROOTS <- c("snapshot", "..")

lv_local_path <- function(table, root) {
  if (table == TRUTH_TABLE) {
    file.path(root, "data", "synthetic-true_parameters.json")
  } else if (table == "MMM_MODEL_RUN") {
    file.path(root, "outputs", "model_metadata.json")
  } else if (table %in% INPUT_TABLES) {
    file.path(root, "data", paste0("synthetic-", tolower(table), ".csv"))
  } else {
    file.path(root, "outputs", paste0(tolower(table), ".csv"))
  }
}

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
lv_connect <- function() {
  # Posit Workbench managed credentials (and Connect service-account
  # credentials) are picked up by odbc::snowflake() automatically.
  DBI::dbConnect(
    odbc::snowflake(),
    account = Sys.getenv("SNOWFLAKE_ACCOUNT", "duloftf-posit-software-pbc-dev"),
    warehouse = LV_WAREHOUSE,
    database = LV_DATABASE,
    schema = LV_SCHEMA
  )
}

lv_source <- local({
  state <- NULL
  function() {
    if (!is.null(state)) return(state)
    mode <- tolower(Sys.getenv("MMM_DATA_SOURCE", "auto"))
    con <- NULL
    error <- NULL
    if (mode != "local") {
      con <- tryCatch(lv_connect(), error = function(e) {
        error <<- conditionMessage(e)
        NULL
      })
      if (is.null(con) && mode == "snowflake") stop("Snowflake connection failed: ", error)
    }
    state <<- list(
      con = con,
      label = if (is.null(con)) "Local files (offline mode)" else paste0("Snowflake · ", LV_DATABASE, ".", LV_SCHEMA),
      error = error
    )
    state
  }
})

# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------
lv_read_local <- function(table, path) {
  if (table == TRUTH_TABLE) {
    channels <- jsonlite::read_json(path)$channels
    return(dplyr::bind_rows(lapply(names(channels), \(ch) c(list(channel = ch), channels[[ch]]))))
  }
  if (table == "MMM_MODEL_RUN") return(tibble::as_tibble(jsonlite::read_json(path)))
  readr::read_csv(path, show_col_types = FALSE)
}

lv_read <- function(table, sql = NULL) {
  src <- lv_source()
  if (!is.null(src$con)) {
    sql <- sql %||% paste0("SELECT * FROM ", LV_DATABASE, ".", LV_SCHEMA, ".", table)
    out <- tryCatch(DBI::dbGetQuery(src$con, sql), error = function(e) {
      # Only "auto" falls back to files, and only for a table that is not there yet.
      if (tolower(Sys.getenv("MMM_DATA_SOURCE", "auto")) == "snowflake" ||
          !grepl("does not exist|not authorized", conditionMessage(e))) stop(e)
      NULL
    })
    # An empty table was created by schema.sql but not loaded (or fitted) yet.
    if (!is.null(out) && (nrow(out) > 0 || table == "MMM_SAVED_SCENARIOS")) {
      names(out) <- tolower(names(out))
      return(tibble::as_tibble(out))
    }
    if (tolower(Sys.getenv("MMM_DATA_SOURCE", "auto")) == "snowflake") {
      stop(LV_DATABASE, ".", LV_SCHEMA, ".", table, " is empty. Load snowflake_setup/ first.")
    }
  }
  for (root in LV_LOCAL_ROOTS) {
    path <- lv_local_path(table, root)
    if (file.exists(path)) return(lv_read_local(table, path))
  }
  if (table == "MMM_SAVED_SCENARIOS") return(tibble::tibble())
  if (table %in% INPUT_TABLES) {
    stop("Synthetic input data not found. Run `uv run python data/generate_data.py` in the project folder.")
  }
  stop("Model output ", table, " not found. Run `uv run python ml/train_model.py` in the project folder.")
}

# Replace the rows of a Snowflake table (types come from snowflake_setup/schema.sql),
# and always keep a local CSV copy.
lv_write <- function(table, df) {
  src <- lv_source()
  if (!is.null(src$con)) {
    upper <- df
    names(upper) <- toupper(names(upper))
    id <- DBI::Id(catalog = LV_DATABASE, schema = LV_SCHEMA, table = table)
    if (!DBI::dbExistsTable(src$con, id)) {
      stop(LV_DATABASE, ".", LV_SCHEMA, ".", table, " does not exist. Run snowflake_setup/schema.sql first.")
    }
    DBI::dbExecute(src$con, paste0("DELETE FROM ", LV_DATABASE, ".", LV_SCHEMA, ".", table))
    DBI::dbAppendTable(src$con, id, upper)
  }
  root <- if (dir.exists("../outputs")) ".." else "snapshot"
  dir.create(file.path(root, "outputs"), showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(df, lv_local_path(table, root))
  invisible(df)
}

# Weekly model inputs: revenue summed over sales channels, spend wide by channel,
# controls. Same shape as load_model_inputs() in mmm_data.py.
lv_model_inputs <- function() {
  sales <- lv_read("SALES_WEEKLY")
  spend <- lv_read("MARKETING_SPEND_WEEKLY")
  controls <- lv_read("CONTROLS_WEEKLY")
  sales |>
    group_by(week_start) |>
    summarise(revenue = sum(revenue), orders = sum(orders), .groups = "drop") |>
    inner_join(
      spend |>
        select(week_start, channel, spend) |>
        tidyr::pivot_wider(names_from = channel, values_from = spend),
      by = "week_start"
    ) |>
    inner_join(controls, by = "week_start") |>
    mutate(week_start = as.Date(week_start)) |>
    arrange(week_start)
}

# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------
lv_brand <- function() {
  path <- if (file.exists("_brand.yml")) "_brand.yml" else "../_brand.yml"
  raw <- yaml::read_yaml(path)
  pal <- raw$color$palette
  roles <- raw$color[setdiff(names(raw$color), "palette")]
  resolved <- lapply(roles, function(v) if (!is.null(pal[[v]])) pal[[v]] else v)
  c(unlist(pal), unlist(resolved))
}

theme_lumenvale <- function(base_size = 11) {
  b <- lv_brand()
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", colour = b[["foreground"]], size = base_size + 2),
      plot.title.position = "plot",
      plot.subtitle = element_text(colour = b[["secondary"]]),
      plot.caption = element_text(colour = b[["secondary"]], hjust = 0),
      axis.text = element_text(colour = b[["secondary"]]),
      axis.title = element_text(colour = b[["secondary"]]),
      panel.grid.major = element_line(colour = "#E8EEF2", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA)
    )
}

lv_money <- function(x, accuracy = 0.01) {
  scales::label_currency(accuracy = accuracy, scale_cut = scales::cut_short_scale())(x)
}
