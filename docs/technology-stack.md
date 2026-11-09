# Technology Stack

Versions below are the declared or lockfile-visible versions. Transitive
versions may vary by lockfile refresh and operating system.

## Python and data science

| Technology | Version | Purpose | Location | Configuration / dependencies |
|---|---:|---|---|---|
| Python | `>=3.11` | Runtime for ingestion, modeling, API, and notebooks | `pyproject.toml`, `src/` | Use `.venv`; package is installed editable with `pip install -e ".[dev]"`. |
| pandas | `>=2.1` | Tabular ingestion, feature construction, reports | `data.py`, `features.py`, `reporting.py`, `dashboard.py` | Reads CSV and Excel through pandas engines. |
| NumPy | `>=1.26` | Numeric operations, safe division, arrays, intervals | Most Python modules | Required by pandas/scikit-learn/CatBoost. |
| scikit-learn | `>=1.4` | Metrics, isotonic calibration, regression/classification evaluation | `validation.py`, `modeling.py` | Uses chronological splits implemented in this repository. |
| CatBoost | `>=1.2` | Categorical-aware gradient boosting regressors/classifiers | `modeling.py` | Thread count defaults to CatBoost’s available cores; can be overridden through optional Settings attributes. |
| openpyxl | `>=3.1` | Excel workbook read/write | `data.py`, `reporting.py`, dashboard previews | Used by pandas Excel readers/writers. |
| xlrd | `>=2.0` | Legacy `.xls` support through pandas | dependency declaration | Needed when users upload older Excel files. |
| pyxlsb | `>=1.0` | `.xlsb` support through pandas | dependency declaration | Needed for binary Excel uploads. |
| joblib | `>=1.3` | Persist model bundles and snapshot dictionaries | `io.py`, `upload_pipeline.py` | Artifacts are Python-version/dependency-sensitive. |
| Jupyter | `>=1.0` | Notebook execution and exploration | `notebooks/` | Notebooks add `src/` to `sys.path`. |
| pytest | `>=8` | Test execution | `tests/`, optional `dev` extra | Run `pytest`. |

## Frontend and build

| Technology | Version | Purpose | Location | Configuration / dependencies |
|---|---:|---|---|---|
| React | `^19.0.0` | Dashboard UI components and state | `dashboard/src/main.jsx` | `react-dom` mounts the app. |
| React DOM | `^19.0.0` | Browser renderer | `dashboard/src/main.jsx` | Entry point calls `createRoot`. |
| Vite | `^7.0.0` | Dev server, proxy, static bundling | `dashboard/vite.config.js` | Proxies `/api` and `/runs` to port 8000; serves generated dashboard data. |
| `@vitejs/plugin-react` | `^5.0.0` | JSX transform and React integration | `dashboard/vite.config.js` | Vite plugin. |
| Node.js | `20+` recommended | Runs Vite and Node utility scripts | dashboard toolchain | `pnpm` is the documented package manager. |
| pnpm | current | Installs frontend dependencies and runs scripts | `dashboard/package.json`, `pnpm-lock.yaml` | Run `pnpm install`, `pnpm dev`, `pnpm build`. |

## Backend and scripts

The backend uses Python’s standard-library `http.server` and
`ThreadingHTTPServer`; there is no Flask, FastAPI, database, queue, or external
service dependency. The dashboard has two Node scripts:

- `dashboard/scripts/clean-assets.cjs` removes the previous compiled assets
  directory within `artifacts/dashboard`.
- `dashboard/scripts/fix-local-html.cjs` makes the built HTML openable from a
  local filesystem by removing module/cross-origin assumptions.

## Data formats and storage

- CSV and Excel (`.xls`, `.xlsx`, `.xlsb`, `.xlsm`) inputs.
- Pickle for cleaned data, predictions, and scoring logs.
- Joblib for snapshot dictionaries and model bundles.
- XLSX for business and model reports.
- JSON and browser JavaScript assignment for dashboard data.
- HTML/CSS/JS for the static dashboard.

## Configuration

`game_performance.config.Settings` is the central default configuration. The
`GAME_DATA_FILE` environment variable overrides the notebook/source workbook
path. The HTTP server supports `--host` and `--port`; Vite supports the usual
Vite environment/configuration conventions and can use `PYTHON` if the local
`.venv` executable is not present.

Model settings include training/validation/holdout fractions, minimum rows,
checkpoint days, forecast horizon, classification threshold, calibration bins,
prediction interval coverage, review thresholds, and training mode budgets.
