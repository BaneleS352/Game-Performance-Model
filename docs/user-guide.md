# User Guide

## Access the platform

From a fresh clone, install Python and dashboard dependencies, then run the
Vite development server from `dashboard/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
cd dashboard
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`. The Vite plugin starts the local Python API as a
child process. If the browser reports proxy errors, confirm that the matching
virtual environment has the Python package installed and restart `pnpm dev`.

## Who can use what

The current local application does not authenticate users. Anyone who can
reach the local process can upload data and view generated reports. Treat the
machine, uploaded workbooks, and generated artifacts as trusted and sensitive.

## Navigation

- **Dashboard**: KPI cards, top systems, top games, and the game ledger.
- **Top Games**: combined ranking using normalised predicted clients plus
  normalised predicted stakes.
- **Top Stakes**: descending predicted seven-day stakes.
- **Top Clients**: descending predicted seven-day clients.
- **Games to Review**: rows where the scoring layer flagged low confidence or
  a wide interval.
- **System filter**: restricts the ledger to one `SystemName`.
- **Search**: case-insensitive search across available forecast columns.
- **Workbook sheets**: opens up to 250 preview rows for visible report sheets.
  Technical sheets—Model Accuracy, Classification Metrics, and Calibration—are
  hidden from the navigation drawer but remain in Model Analysis.xlsx.
- **New Workbook**: returns to the upload workflow.

## Upload and score a workbook

1. Choose **New Workbook**.
2. Select a `.csv`, `.xls`, `.xlsx`, `.xlsb`, or `.xlsm` file no larger than 65
   MB.
3. Wait for the workbook to upload and select a worksheet.
4. Confirm the worksheet contains the required headers:
   `Date`, `SystemName`, `TournamentName`, `Clients`, `BetCount`, `Stakes`,
   and `GGR`. Header casing is accepted case-insensitively.
5. Click **Run production model** once.
6. Leave the page open or return later; the job id is persisted in browser local
   storage and the page resumes polling after refresh.
7. When complete, inspect the dashboard and download **Business Report.xlsx**
   and **Model Analysis.xlsx**.

## Required input behavior

- `Date` must parse as a date.
- `SystemName` and `TournamentName` must be present and non-blank.
- `Clients`, `BetCount`, and `Stakes` must be numeric and non-negative.
- `GGR` must be numeric; negative GGR is allowed because it can represent a
  negative margin.
- Duplicate `(Date, SystemName, TournamentName)` records are de-duplicated by
  keeping the last valid source occurrence.
- The worksheet should contain enough history for at least one configured
  checkpoint and enough rows for temporal training/validation requirements.

## Interpreting the result

- Forecast values are for the next seven calendar days after the observation
  end date, shown by `Prediction_Start` and `Prediction_End`.
- Low/high fields are prediction bands, not guarantees.
- Confidence is derived from validation error and/or prediction-bin confidence
  tables; lower values require more caution.
- `predicted_high_client_performer` and `predicted_high_stakes_performer` are
  separate goal flags. `predicted_high_performer` requires both.
- `requires_review` is true when clients or stakes confidence is below the
  configured threshold or the relative interval width is too wide.
- Recommendations are heuristic merchandising labels, not an automatic change
  to a catalogue or promotion system.

## Common errors and resolution

| Message / symptom | Likely cause | Resolution |
|---|---|---|
| `Upload a CSV or Excel file...` | Unsupported extension | Save as CSV/XLS/XLSX/XLSB/XLSM and retry. |
| `The workbook must be 65 MB or smaller.` | Request exceeds API limit | Reduce workbook size or select only the needed worksheet. |
| `Missing required source columns` | Header absent or misspelled | Rename the seven required columns; casing is not significant. |
| `Worksheet not found` | Selection changed or stale job | Upload the workbook again and choose an existing sheet. |
| `The selected worksheet has no valid rows` | Dates, identifiers, or numeric values failed cleaning | Inspect blanks, text-formatted numbers, negative activity, and invalid dates. |
| `No production model families could be trained` | Too few eligible rows/history or no valid split | Supply more chronological history and enough unique launch dates. |
| `No games are eligible to score yet` | Games have not reached the minimum configured checkpoint | Wait for more observed days or use a model trained for an earlier checkpoint. Notebook 04 may prefer a 14-day pair only when that family exists. |
| Proxy error for `/api/...` | Python API did not start | Install the package in `.venv`, check the terminal, and restart `pnpm dev`. |
| Dashboard data is missing | No successful scoring run has published data | Upload and run a workbook, or run `python -m game_performance.dashboard` against a report. |

## User responsibilities

Users must select the correct worksheet, validate that the source extract is
complete, avoid uploading confidential data to an untrusted machine, review
uncertainty flags, and avoid treating a model recommendation as a fully
automated business decision.
