# `scoring.py`

## Purpose

Runs current lifecycle inference and converts model outputs into operational
forecast rows, confidence, review flags, and recommendations.

## Key functions

- `score_snapshot`: scores one snapshot using driver or backward-compatible
  direct-outcome bundles.
- `eligible_checkpoint_for_age`: chooses exact or latest eligible checkpoint.
- `run_daily_lifecycle_scoring`: scores eligible games, appends a deduplicated
  scoring log, and returns predictions plus log.
- `assign_merchandising_recommendation`: maps probability/activity to a label.

## Inputs and outputs

Inputs are cleaned data, model bundles, a checkpoint/horizon strategy, and an
optional as-of date/log. Outputs include point forecasts, low/high bands,
confidence, calibrated probabilities, performer flags, review reason/priority,
recommendations, lineage, and timestamps.

## Errors and related modules

Missing bundle keys or no eligible games cause errors upstream or empty outputs.
Uses features and modeling helpers; feeds reporting, dashboard, and upload
pipeline.
