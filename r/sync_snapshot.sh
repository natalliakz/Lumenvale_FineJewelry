#!/usr/bin/env bash
# Copy the brand file and a local snapshot of the data into r/ so the R reports
# can be deployed to Posit Connect without Snowflake (offline fallback).
# With Snowflake configured on Connect, only _brand.yml is needed.
#
# DISCLAIMER: This project contains synthetic data and analysis created for
# demonstration purposes only.
set -euo pipefail
cd "$(dirname "$0")"

cp ../_brand.yml _brand.yml
mkdir -p snapshot/data snapshot/outputs
cp ../data/synthetic-*.csv ../data/synthetic-true_parameters.json snapshot/data/
cp ../outputs/*.csv ../outputs/model_metadata.json snapshot/outputs/
echo "Snapshot written to r/snapshot/ ($(ls snapshot/data snapshot/outputs | grep -c csv) CSV files)."
