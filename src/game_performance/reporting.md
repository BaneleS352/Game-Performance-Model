# `reporting.py`

## Purpose

Creates business-friendly holdout comparisons and writes Business Report and
Model Analysis workbooks.

## Key functions

- `build_holdout_comparison`: adds actual/baseline/predicted variance,
  probability, off-by-10/20, range, and performer fields.
- `build_forecast_capability_report`: summarises history/horizon, games, WAPE,
  and range coverage.
- `build_calibration_report`: buckets probability against observed hit rate.
- `export_business_report`: writes the concise five-sheet workbook.
- `export_excel`: writes the full technical workbook.

## Inputs, outputs, errors

Inputs are holdout predictions, current forecasts, evaluations, calibration,
selection, uncertainty, and impact frames. Outputs are XLSX files with frozen
headers and bounded column widths. Empty inputs produce empty sheets where
possible; Excel/path errors propagate. Uses pandas, NumPy, openpyxl, and
internal validation/impact/features modules.
