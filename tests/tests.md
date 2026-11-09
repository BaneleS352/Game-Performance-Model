# `tests/` Folder Guide

## Folder purpose

Pytest regression and behavior tests for the reusable package and dashboard/API
workflow.

## File inventory

| File | Coverage |
|---|---|
| `test_data.py` | Schema, cleaning, identifiers, numeric parsing, duplicates. |
| `test_features.py` | Lifecycle windows, features, targets, leakage behavior. |
| `test_validation.py` | Temporal splits, folds, target thresholds, metrics. |
| `test_modeling.py` | Model helpers, transformations, intervals, calibration, training contracts. |
| `test_scoring.py` | Forecast outputs, confidence, review flags, recommendations, daily scoring. |
| `test_reporting.py` | Holdout calculations, calibration, Excel sheet schemas, impact exports. |
| `test_impact.py` | External-impact detection and forecast flags. |
| `test_dashboard.py` | Dashboard serialization, metadata, latest checkpoints, previews. |
| `test_upload_pipeline.py` | Upload safety, size limits, job state, pipeline stage order, failure paths. |
| `tests.md` | Folder guide | Markdown | Describes test ownership, execution, and coverage. |

## Execution order

Tests are independent fixtures except where they deliberately exercise package
contracts. Run `python -m pytest` from the repository root. The package is made
available through `tool.pytest.ini_options.pythonpath = ["src"]`.

## Ownership

Every change to source schema, output fields, API behavior, or model bundle
structure should add or update the relevant test before merging.
