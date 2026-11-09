# `impact.py`

## Purpose

Infers likely external-impact periods from unusual metric movement and attaches
compact context to forecasts.

## Inputs and outputs

`detect_external_impact_periods` compares each metric to a rolling seven-day
median baseline after a minimum history of three rows; changes of at least 50%
are grouped into contiguous periods. It returns `IMPACT_PERIOD_COLUMNS`.
`add_external_impact_flags` matches periods to forecasts within a 14-day lookback
and adds detection, period, and reason fields.

## Interpretation and errors

The output is an inference, not causal proof or a promotion ledger. Missing
required columns raise `KeyError`; empty input returns an empty schema. Uses
pandas and NumPy; feeds reporting and upload orchestration.
