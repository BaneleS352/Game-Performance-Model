# Game Performance ML — Refactored Project

This project separates the original monolithic lifecycle-model notebook into thin orchestration notebooks and reusable Python modules.

## Structure

```text
notebooks/
  01_data_exploration.ipynb
  02_feature_engineering.ipynb
  03_model_experiments.ipynb
  04_training_and_forecasting.ipynb
src/game_performance/
  config.py       # environment-driven paths and ML settings
  data.py         # ingestion, schema validation, cleaning, EDA summaries
  features.py     # point-in-time lifecycle features and model matrices
  validation.py   # temporal split, rolling folds, leakage-safe targets, metrics
  modeling.py     # CatBoost tuning and model-family training
  scoring.py      # current-game scoring and recommendations
  reporting.py    # holdout comparison, KPIs, Excel export
  io.py           # explicit artifact persistence
  tests/
```

This repository intentionally excludes source workbooks, uploaded data, trained
models, reports, and generated dashboard data. A fresh clone starts with a blank
dashboard; upload your own workbook and run the production model to create local
outputs.

## Quick start from a fresh clone

Prerequisites:

- Python 3.11 or newer
- Node.js 20 or newer
- `pnpm`

```bash
git clone <repo-url>
cd Game-Performance-Model
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"

cd dashboard
pnpm install
pnpm dev
```

macOS/Linux:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

cd dashboard
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`, upload your workbook, choose a worksheet, and run
the production model. The dev server also starts the local Python API used for
uploads and scoring.

Activate the virtual environment first if you prefer using plain `python`:

```bash
source .venv/bin/activate
```

In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the browser shows proxy errors for `/api/dashboard-data` or `/api/uploads`,
the Python API did not start. Install the Python dependencies with the matching
`.venv` command above, then restart `pnpm dev` from `dashboard/`.

To build a static dashboard after data has been generated:

```bash
cd dashboard
pnpm build
```

Then open `artifacts/dashboard/index.html`.

## Dashboard

The dashboard is a React/Vite static app that reads the exported workbook data
from `artifacts/dashboard/dashboard_data.js`.

```bash
python -m game_performance.dashboard
cd dashboard
pnpm install
pnpm build
```

Open `artifacts/dashboard/index.html` after building. The dashboard shows
curated Top Stakes and Top Clients ledger views with focused forecast ranges.

The private source workbook and generated outputs are not included in the repository. Static compilation and unit tests validate the refactored code paths; each user must train a model from their own uploaded data.

### Score an uploaded workbook locally

Start the Vite dashboard:

```bash
python -m pip install -e ".[dev]"
cd dashboard
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`, choose a CSV or Excel file (`.csv`, `.xls`, `.xlsx`, `.xlsb`, or `.xlsm`) and worksheet, then run the full production model. The selected sheet must include `Date`, `SystemName`, `TournamentName`, `Clients`, `BetCount`, `Stakes`, and `GGR` columns; casing is accepted case-insensitively. The upload runs the same four notebook stages through the reusable `game_performance` modules: data cleaning, lifecycle feature engineering, full regression and classification training, then forecasting, reporting, and dashboard publishing. Each upload and generated report is kept under `data/uploads/` and `artifacts/uploads/`; the latest successful run also publishes the processed data, features, model artifact, and dashboard.

Each run provides two downloads: `Business Report.xlsx` with a forecast-capability summary, concise holdout results, high performers, and forecasts; and `Model Analysis.xlsx` with the complete model outputs. The `Forecast Capability` sheet states how many daily history days were used, the derived variable families, the forecast period, the number of holdout games evaluated, WAPE for Clients, Stakes, BetCount, and GGR, their average percentage error, and prediction-range coverage.

Uploaded data is split chronologically by unique launch date: 70% for training,
15% for validation, and 15% for holdout scoring. For example, 60 dates produce
42 training dates, 9 validation dates, and 9 holdout dates.

After that evaluation, every forecast covers the next seven calendar days after
the final observed date. `Prediction_Start` and `Prediction_End` show that
window; forecasts are not labelled or calculated as lifecycle week-four values.
