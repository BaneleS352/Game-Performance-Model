# `artifacts/features/` Folder Guide

## Folder purpose

Stores the serialized historical lifecycle snapshot mapping used by training.

## File inventory

| File | Purpose |
|---|---|
| `historical_snapshots.joblib` | `(checkpoint, horizon) -> DataFrame` mapping. |
| `features.md` | Folder guide. |

Produced by notebook 02 and the upload pipeline; consumed by modeling.
