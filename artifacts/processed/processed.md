# `artifacts/processed/` Folder Guide

## Folder purpose

Stores the latest canonical cleaned daily dataset.

## File inventory

| File | Purpose |
|---|---|
| `clean_game_daily.pkl` | Cleaned DataFrame with stable game/lifecycle fields and audit attributes. |
| `processed.md` | Folder guide. |

Produced by notebook 01 or the upload pipeline; consumed by feature engineering,
scoring, and reports.
