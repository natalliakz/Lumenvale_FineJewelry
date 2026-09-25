"""Create LUMENVALE_MMM.PUBLIC and load every demo table, from Posit Workbench.

Runs schema.sql, then inserts the same data as the CSV bundle: the synthetic
inputs, the generator's answer key, the model outputs and the R cross-check.
Uses Workbench-managed Snowflake credentials (connection "workbench").

    uv run python snowflake_setup/load_to_snowflake.py               # everything
    uv run python snowflake_setup/load_to_snowflake.py --inputs-only # then refit in Snowflake:
    uv run python ml/train_model.py

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import argparse

from build_bundle import BUNDLED_TABLES, SCHEMA_SQL, table_frames  # also puts the project on sys.path

from mmm_data import DATABASE, INPUT_TABLES, SCHEMA, TRUTH_TABLE, connect, write_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs-only", action="store_true",
                        help="load only the input tables; refit the model to fill the rest")
    args = parser.parse_args()

    tables = INPUT_TABLES + (TRUTH_TABLE,) if args.inputs_only else BUNDLED_TABLES
    frames = table_frames(tables)
    con = connect()
    for cur in con.execute_string(SCHEMA_SQL.read_text(), remove_comments=True):
        print(f"  ok: {cur.query.split(chr(10))[0][:72]}")
    for table, df in frames.items():
        # Columns are upper-case already; write_table upper-cases the INSERT list anyway.
        rows = write_table(con, table, df.rename(columns=str.lower), overwrite=True)
        print(f"  loaded {rows:>6,} rows into {DATABASE}.{SCHEMA}.{table}")
    con.close()


if __name__ == "__main__":
    main()
