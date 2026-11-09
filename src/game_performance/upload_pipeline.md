# `upload_pipeline.py`

## Purpose

Orchestrates an uploaded worksheet through cleaning, impact detection, feature
engineering, production training, scoring, reporting, dashboard export, and
atomic publication.

## Key functions

- `safe_workbook_filename`: extension allow-list and safe stem.
- `save_uploaded_workbook`: writes bytes under `data/uploads/` with a UTC id.
- `workbook_sheets`: returns available worksheets.
- `run_full_production_pipeline`: runs all stages and returns run/artifact/summary metadata.
- `score_uploaded_workbook`: reuses an existing model artifact for scoring only.

## Inputs and outputs

Inputs are project root, workbook path, worksheet name, optional progress
callback/model path. Outputs include run-scoped and latest artifacts, reports,
dashboard data, cleaning summary, and training metadata.

## Error handling

Invalid extension, empty upload, missing worksheet/model, empty/invalid data,
insufficient model families, or no eligible games raise clear exceptions. Live
artifacts are published only after the complete run succeeds. Uses pandas and
all core package modules.
