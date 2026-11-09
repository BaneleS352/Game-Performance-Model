# `01_data_exploration.ipynb`

## Purpose and location

Ingest, inspect, clean, validate, and persist the canonical daily dataset.

## Inputs

The workbook configured by `GAME_DATA_FILE` or the notebook’s `SOURCE_FILE`
cell. The first worksheet is loaded after listing available sheets.

## Outputs

`artifacts/processed/clean_game_daily.pkl` plus displayed quality summaries.

## Dependencies and process

Uses `Settings.ensure_directories`, `load_raw_workbook`, `clean_game_daily`,
`data_quality_summary`, and `save_dataframe`. It reports row counts, dates,
missing values, numeric summaries, and cleaning audit attributes.

## Errors and example

Missing source files raise `FileNotFoundError`; invalid schemas raise
`ValueError`. Run the notebook after setting `GAME_DATA_FILE` and before
`02_feature_engineering.ipynb`.
