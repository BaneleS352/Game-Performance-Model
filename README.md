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

The raw workbook is not included in this refactor, so full model training was not executed here. Static compilation and unit tests validate the refactored code paths that do not require the private source workbook.

### Score an uploaded workbook locally

Start the Vite dashboard:

```bash
python -m pip install -e ".[dev]"
cd dashboard
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`, choose a CSV or Excel file (`.csv`, `.xls`, `.xlsx`, `.xlsb`, or `.xlsm`) and worksheet, then run the existing trained model. The selected sheet must include `Date`, `SystemName`, `TournamentName`, `Clients`, and `Stakes` columns. Each upload and generated report is kept under `data/uploads/` and `artifacts/uploads/`; the dashboard shows the latest successful run.
