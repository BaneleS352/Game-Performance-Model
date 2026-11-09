# `data/` Folder Guide

## Folder purpose

Runtime input storage. Uploaded workbooks are kept under `data/uploads/` so a
run can be reproduced or investigated.

## File inventory

| Path | Purpose | Dependencies | Description |
|---|---|---|---|
| `uploads/*.csv` / `uploads/*.xls*` | User inputs | Upload API, pandas Excel engines | Timestamped/sanitised workbook copies. |
| `data.md` | Folder guide | Markdown | Describes data storage and retention. |

## Key processes

`server.py` calls `save_uploaded_workbook`; `upload_pipeline.py` loads a selected
worksheet and writes cleaned/runtime outputs under `artifacts/`. Source files
are not transformed in place.

## Data dependencies and ownership

The input contract is documented in [Data and reporting](../docs/data-and-reporting.md).
Uploaded files may contain sensitive commercial data; the operating owner
controls retention and access.
