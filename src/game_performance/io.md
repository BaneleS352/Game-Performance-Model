# `io.py`

## Purpose

Small explicit persistence boundary for DataFrames and arbitrary model
artifacts.

## Key functions

- `save_dataframe` / `load_dataframe`: pickle DataFrames.
- `save_artifact` / `load_artifact`: joblib-dump/load arbitrary objects such as
  snapshot dictionaries and model bundles.

## Inputs, outputs, errors

Save functions create parent directories and return the target `Path`; load
functions return the deserialised object. Missing paths or incompatible/corrupt
artifacts raise the underlying filesystem/joblib exception.

## Dependencies and related modules

Uses pandas and joblib. Called by notebooks and `upload_pipeline.py`.
