# `data.py`

## Purpose

Loads source workbooks, validates required headers, cleans daily records, and
returns quality summaries.

## Inputs and outputs

`load_raw_workbook(path, sheet_name)` accepts CSV or Excel paths and returns a
DataFrame. `clean_game_daily(df)` returns canonical rows with `Game_ID`,
lifecycle dates, and calculated GGR percentage; cleaning counts are stored in
`DataFrame.attrs`. `data_quality_summary` returns shape, missing, numeric,
date-range, and audit tables.

## Processing and errors

Headers are case-insensitive; duplicate required headers fail. Dates and numeric
text are coerced, invalid/negative activity rows are removed by default, and
duplicate daily game rows keep the last valid occurrence. Missing files raise
`FileNotFoundError`; bad schema/policy raises `ValueError`.

## Dependencies and related modules

Uses pandas and NumPy. Feeds `features.py`, `impact.py`, `upload_pipeline.py`,
and notebook 01.
