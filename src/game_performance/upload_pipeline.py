"""Run uploaded workbooks through the production model workflow."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import Settings
from .dashboard import export_dashboard
from .data import clean_game_daily, load_raw_workbook
from .features import build_all_historical_snapshots
from .io import load_artifact, save_artifact, save_dataframe
from .impact import add_external_impact_flags, detect_external_impact_periods
from .modeling import train_all_model_families
from .reporting import build_holdout_comparison, export_business_report, export_excel
from .scoring import run_daily_lifecycle_scoring

SUPPORTED_UPLOAD_EXTENSIONS = {".csv", ".xls", ".xlsx", ".xlsb", ".xlsm"}
LOGGER = logging.getLogger(__name__)


def safe_workbook_filename(filename: str) -> str:
    """Return a safe spreadsheet filename suitable for local storage."""
    name = Path(filename).name.strip()
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Upload a CSV or Excel file (.csv, .xls, .xlsx, .xlsb, or .xlsm).")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    if not stem:
        raise ValueError("The workbook filename must contain letters or numbers.")
    return f"{stem}{suffix}"


def save_uploaded_workbook(
    project_root: str | Path,
    filename: str,
    contents: bytes,
) -> tuple[str, Path]:
    """Save an uploaded spreadsheet and return its id and path."""
    if not contents:
        raise ValueError("The uploaded spreadsheet is empty.")

    safe_name = safe_workbook_filename(filename)
    upload_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    upload_dir = Path(project_root) / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{upload_id}_{safe_name}"
    path.write_bytes(contents)
    return upload_id, path


def workbook_sheets(path: str | Path) -> list[str]:
    """Return worksheet names, using one sheet for CSV files."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return [path.stem]
    return pd.ExcelFile(path).sheet_names


def _selected_strategy(results: dict[str, Any]) -> tuple[dict[int, list[int]], int | None, int | None]:
    bundles = results.get("model_bundles")
    if not isinstance(bundles, dict) or not bundles:
        raise ValueError("The trained model artifact does not contain model bundles.")

    selection = results.get("selection_evaluation", {})
    if isinstance(selection, dict) and {"Checkpoint", "Horizon"}.issubset(selection):
        checkpoint = int(selection["Checkpoint"])
        horizon = int(selection["Horizon"])
        if (checkpoint, horizon) in bundles:
            return {checkpoint: [horizon]}, checkpoint, horizon

    strategy: dict[int, list[int]] = {}
    for checkpoint, horizon in bundles:
        strategy.setdefault(int(checkpoint), []).append(int(horizon))
    return strategy, None, None


def _publish_dashboard(staged_dir: Path, dashboard_dir: Path) -> Path:
    """Atomically publish dashboard data only after the complete run succeeds."""
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dashboard_data.json", "dashboard_data.js"):
        source = staged_dir / name
        target = dashboard_dir / name
        temporary = target.with_name(f".{name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    return dashboard_dir / "dashboard_data.json"


def _publish_file(source: Path, target: Path) -> None:
    """Publish one completed artifact without exposing a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _progress(
    callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(fraction, message)


def run_full_production_pipeline(
    project_root: str | Path,
    workbook_path: str | Path,
    sheet_name: str,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run notebook stages 01-04 against an uploaded worksheet.

    The notebooks are orchestration layers; their production work is performed
    here by the same reusable data, feature, modeling, scoring, and reporting
    modules used by the project.
    """
    root = Path(project_root)
    workbook_path = Path(workbook_path)
    started_at = perf_counter()
    if sheet_name not in workbook_sheets(workbook_path):
        raise ValueError(f"Worksheet not found: {sheet_name}")

    settings = Settings(project_root=root)
    settings.ensure_directories()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = root / "artifacts" / "uploads" / run_id

    LOGGER.info("Processing uploaded workbook: %s", workbook_path.name)
    _progress(progress_callback, 0.05, "Validating and cleaning uploaded data")
    try:
        cleaned = clean_game_daily(load_raw_workbook(workbook_path, sheet_name=sheet_name))
    except Exception:
        LOGGER.exception("Upload validation or cleaning failed for %s", workbook_path.name)
        raise
    if cleaned.empty:
        raise ValueError("The selected worksheet has no valid rows to score.")
    cleaning_summary = dict(cleaned.attrs)
    LOGGER.info(
        "Upload cleaned: rows=%s duplicates_removed=%s invalid_rows=%s missing_values=%s outliers=%s",
        cleaning_summary["output_rows"],
        cleaning_summary["duplicate_rows_removed"],
        cleaning_summary["invalid_rows"],
        cleaning_summary["missing_values_found"],
        cleaning_summary["outliers_detected"],
    )
    impact_periods = detect_external_impact_periods(cleaned)
    uploaded_data_days = int(cleaned["Date"].dropna().nunique())
    cleaned_path = save_dataframe(cleaned, run_dir / "clean_game_daily.pkl")

    _progress(progress_callback, 0.18, "Building lifecycle features")
    historical_snapshots = build_all_historical_snapshots(cleaned, settings.horizon_strategy)
    features_path = save_artifact(historical_snapshots, run_dir / "historical_snapshots.joblib")

    _progress(progress_callback, 0.28, "Training all production model families")
    training_results = train_all_model_families(
        historical_snapshots,
        settings,
        mode="production",
        max_families=0,
        run_regression=True,
        run_classification=True,
        verbose=True,
    )
    model_bundles = training_results.get("model_bundles")
    if not isinstance(model_bundles, dict) or not model_bundles:
        skipped = training_results.get("skipped_configurations")
        detail = ""
        if isinstance(skipped, pd.DataFrame) and not skipped.empty:
            reasons = "; ".join(
                f"day {row.Checkpoint} / horizon {row.Horizon}: {row.Reason}"
                for row in skipped.itertuples(index=False)
            )
            detail = f" Skipped configurations: {reasons}."
        raise ValueError(f"No production model families could be trained.{detail}")
    training_results.setdefault("training_metadata", {})["pipeline_stages"] = [
        "01_data_exploration",
        "02_feature_engineering",
        "03_model_experiments",
        "04_training_and_forecasting",
    ]
    training_results["training_metadata"].update(
        {
            "uploaded_data_days": uploaded_data_days,
            "temporal_split_fractions": {"training": 0.70, "validation": 0.15, "holdout": 0.15},
        }
    )
    model_path = save_artifact(training_results, run_dir / "training_results.joblib")

    _progress(progress_callback, 0.76, "Generating forecasts and business outputs")
    scoring_strategy: dict[int, list[int]] = {}
    for checkpoint, horizon in model_bundles:
        scoring_strategy.setdefault(int(checkpoint), []).append(int(horizon))
    forecasts, scoring_log = run_daily_lifecycle_scoring(
        cleaned,
        model_bundles,
        scoring_strategy,
        as_of_date=cleaned["Date"].max(),
        exact_checkpoint_only=False,
    )
    forecasts = add_external_impact_flags(forecasts, impact_periods)
    if forecasts.empty:
        minimum_checkpoint = min(checkpoint for checkpoint, _ in model_bundles)
        raise ValueError(
            "No games are eligible to score yet. The production model requires at least "
            f"{minimum_checkpoint} observed days."
        )

    save_dataframe(forecasts, run_dir / "current_predictions.pkl")
    save_dataframe(scoring_log, run_dir / "scoring_log.pkl")
    holdout_predictions = training_results.get("holdout_predictions", pd.DataFrame())
    comparison = pd.DataFrame()
    if isinstance(holdout_predictions, pd.DataFrame) and not holdout_predictions.empty:
        available_pairs = sorted(
            {
                (int(row.Obs_Window), int(row.Horizon))
                for row in holdout_predictions[["Obs_Window", "Horizon"]].drop_duplicates().itertuples(index=False)
            }
        )
        preferred_pair = (settings.representative_tuning_checkpoint, settings.forecast_horizon)
        checkpoint, horizon = preferred_pair if preferred_pair in available_pairs else available_pairs[-1]
        comparison = build_holdout_comparison(
            holdout_predictions,
            checkpoint=checkpoint,
            horizon=horizon,
        )

    business_report_path = run_dir / "Business Report.xlsx"
    model_analysis_path = run_dir / "Model Analysis.xlsx"
    export_business_report(
        business_report_path,
        comparison=comparison,
        forecasts=forecasts,
        impact_periods=impact_periods,
    )
    export_excel(
        model_analysis_path,
        comparison=comparison,
        forecasts=forecasts,
        regression=training_results.get("regression_evaluation", pd.DataFrame()),
        classification=training_results.get("classification_evaluation", pd.DataFrame()),
        calibration=training_results.get("classification_calibration"),
        configuration_leaderboard=training_results.get("configuration_leaderboard"),
        uncertainty_validation=training_results.get("uncertainty_validation"),
        impact_periods=impact_periods,
    )
    staged_dashboard = run_dir / "dashboard"
    staged_data_path = export_dashboard(model_analysis_path, staged_dashboard)
    dashboard_data = json.loads(staged_data_path.read_text(encoding="utf-8"))

    _progress(progress_callback, 0.94, "Publishing production artifacts")
    _publish_file(cleaned_path, settings.processed_dir / "clean_game_daily.pkl")
    _publish_file(features_path, settings.feature_dir / "historical_snapshots.joblib")
    _publish_file(model_path, settings.model_dir / "training_results.joblib")
    dashboard_path = _publish_dashboard(staged_dashboard, root / "artifacts" / "dashboard")
    _progress(progress_callback, 1.0, "Production run complete")
    LOGGER.info("Upload processing complete in %.2fs", perf_counter() - started_at)

    return {
        "run_id": run_id,
        "sheet_name": sheet_name,
        "report_path": str(business_report_path),
        "business_report_path": str(business_report_path),
        "model_analysis_path": str(model_analysis_path),
        "dashboard_data_path": str(dashboard_path),
        "model_path": str(root / "artifacts" / "models" / "training_results.joblib"),
        "summary": dashboard_data["summary"],
        "cleaning_summary": cleaning_summary,
        "training_metadata": training_results.get("training_metadata", {}),
    }


def score_uploaded_workbook(
    project_root: str | Path,
    workbook_path: str | Path,
    sheet_name: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score one uploaded sheet with an existing trained artifact."""
    root = Path(project_root)
    workbook_path = Path(workbook_path)
    LOGGER.info("Scoring uploaded workbook: %s", workbook_path.name)
    if sheet_name not in workbook_sheets(workbook_path):
        raise ValueError(f"Worksheet not found: {sheet_name}")

    cleaned = clean_game_daily(load_raw_workbook(workbook_path, sheet_name=sheet_name))
    if cleaned.empty:
        raise ValueError("The selected worksheet has no valid rows to score.")
    cleaning_summary = dict(cleaned.attrs)
    impact_periods = detect_external_impact_periods(cleaned)

    model_path = Path(model_path or root / "artifacts" / "models" / "training_results.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found: {model_path}")
    results = load_artifact(model_path)
    if not isinstance(results, dict):
        raise ValueError("The trained model artifact has an invalid format.")

    strategy, checkpoint, horizon = _selected_strategy(results)
    forecasts, scoring_log = run_daily_lifecycle_scoring(
        cleaned,
        results["model_bundles"],
        strategy,
        as_of_date=cleaned["Date"].max(),
        exact_checkpoint_only=False,
    )
    forecasts = add_external_impact_flags(forecasts, impact_periods)
    if forecasts.empty:
        minimum_checkpoint = min(strategy)
        raise ValueError(
            f"No games are eligible to score yet. The trained model requires at least "
            f"{minimum_checkpoint} observed days."
        )

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = root / "artifacts" / "uploads" / run_id
    business_report_path = run_dir / "Business Report.xlsx"
    model_analysis_path = run_dir / "Model Analysis.xlsx"
    save_dataframe(cleaned, run_dir / "clean_game_daily.pkl")
    save_dataframe(forecasts, run_dir / "current_predictions.pkl")
    save_dataframe(scoring_log, run_dir / "scoring_log.pkl")

    holdout_predictions = results.get("holdout_predictions", pd.DataFrame())
    comparison = (
        build_holdout_comparison(holdout_predictions, checkpoint=checkpoint, horizon=horizon)
        if isinstance(holdout_predictions, pd.DataFrame) and "Obs_Window" in holdout_predictions
        else pd.DataFrame()
    )
    export_business_report(
        business_report_path,
        comparison=comparison,
        forecasts=forecasts,
        impact_periods=impact_periods,
    )
    export_excel(
        model_analysis_path,
        comparison=comparison,
        forecasts=forecasts,
        regression=results.get("regression_evaluation", pd.DataFrame()),
        classification=results.get("classification_evaluation", pd.DataFrame()),
        calibration=results.get("classification_calibration", pd.DataFrame()),
        configuration_leaderboard=results.get("configuration_leaderboard"),
        impact_periods=impact_periods,
    )

    staged_dashboard = run_dir / "dashboard"
    staged_data_path = export_dashboard(model_analysis_path, staged_dashboard)
    dashboard_data = json.loads(staged_data_path.read_text(encoding="utf-8"))
    dashboard_path = _publish_dashboard(staged_dashboard, root / "artifacts" / "dashboard")

    return {
        "run_id": run_id,
        "sheet_name": sheet_name,
        "report_path": str(business_report_path),
        "business_report_path": str(business_report_path),
        "model_analysis_path": str(model_analysis_path),
        "dashboard_data_path": str(dashboard_path),
        "summary": dashboard_data["summary"],
        "cleaning_summary": cleaning_summary,
    }
