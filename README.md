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

## Key corrections from the original notebook

1. **Fixed destructive numeric cleaning order.** Currency/percentage text is stripped before numeric coercion; formatted values are no longer converted to `NaN` prematurely.
2. **Removed `globals().update(...)`.** Every notebook loads named artifacts explicitly, making dependencies inspectable and reproducible.
3. **Moved reusable functions out of notebooks.** Feature engineering, validation, modeling, scoring, and reporting now have single sources of truth.
4. **Strengthened leakage controls.** Holdout target scoring uses distributions and missing-value medians learned from development data only.
5. **Separated live features from future targets.** `require_future=False` never emits `target_*` columns.
6. **Replaced hard-coded Windows source path.** Set `GAME_DATA_FILE` or use the default `data/raw/game_performance.xlsx`.
7. **Added schema checks, tests, package metadata, and deterministic settings.**
8. **Added accuracy-first lifecycle selection.** The pipeline compares checkpoints at days 21, 28, 35, 42, 49, and 56 for the next seven days, using rolling temporal validation and recent-versus-expanding training windows.

## Execution order

```bash
python -m pip install -e .
export GAME_DATA_FILE=/path/to/Tournament_Daily_Dataset.xlsx
jupyter lab
```

Run notebooks in numeric order. Each stage writes an explicit artifact used by the next stage.

## Artifact contract

- `artifacts/processed/clean_game_daily.pkl`
- `artifacts/features/historical_snapshots.joblib`
- `artifacts/models/training_results.joblib`
- `artifacts/reports/current_predictions.pkl`
- `artifacts/reports/model_business_report.xlsx`

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

In a second terminal from the project root:

```bash
python -m game_performance.server
```

Open `http://127.0.0.1:5173`, choose an `.xlsx` workbook and worksheet, then run the existing trained model. The selected sheet must include `Date`, `SystemName`, `TournamentName`, `Clients`, and `Stakes` columns. Each upload and generated report is kept under `data/uploads/` and `artifacts/uploads/`; the dashboard shows the latest successful run.
