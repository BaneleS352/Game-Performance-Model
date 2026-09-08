"""Score uploaded workbooks with the existing trained model."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .dashboard import export_dashboard
from .data import clean_game_daily, load_raw_workbook
from .io import load_artifact, save_dataframe
from .reporting import build_holdout_comparison, export_excel
from .scoring import run_daily_lifecycle_scoring


def safe_workbook_filename(filename: str) -> str:
    """Return a safe .xlsx filename suitable for local storage."""
    name = Path(filename).name.strip()
    if Path(name).suffix.lower() != ".xlsx":
        raise ValueError("Upload an Excel workbook with a .xlsx extension.")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    if not stem:
        raise ValueError("The workbook filename must contain letters or numbers.")
    return f"{stem}.xlsx"


def save_uploaded_workbook(
    project_root: str | Path,
    filename: str,
    contents: bytes,
) -> tuple[str, Path]:
    """Save an uploaded .xlsx workbook and return its id and path."""
    if not contents:
        raise ValueError("The uploaded workbook is empty.")
    if not zipfile.is_zipfile(BytesIO(contents)):
        raise ValueError("The uploaded file is not a valid .xlsx workbook.")

    safe_name = safe_workbook_filename(filename)
    upload_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    upload_dir = Path(project_root) / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{upload_id}_{safe_name}"
    path.write_bytes(contents)
    return upload_id, path


def workbook_sheets(path: str | Path) -> list[str]:
    """Return the worksheet names available to score."""
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


def score_uploaded_workbook(
    project_root: str | Path,
    workbook_path: str | Path,
    sheet_name: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Clean, score, report, and publish dashboard data for one uploaded sheet."""
    root = Path(project_root)
    workbook_path = Path(workbook_path)
    if sheet_name not in workbook_sheets(workbook_path):
        raise ValueError(f"Worksheet not found: {sheet_name}")

    cleaned = clean_game_daily(load_raw_workbook(workbook_path, sheet_name=sheet_name))
    if cleaned.empty:
        raise ValueError("The selected worksheet has no valid rows to score.")

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
    if forecasts.empty:
        minimum_checkpoint = min(strategy)
        raise ValueError(
            f"No games are eligible to score yet. The trained model requires at least "
            f"{minimum_checkpoint} observed days."
        )

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = root / "artifacts" / "uploads" / run_id
    report_path = run_dir / "model_business_report.xlsx"
    save_dataframe(cleaned, run_dir / "clean_game_daily.pkl")
    save_dataframe(forecasts, run_dir / "current_predictions.pkl")
    save_dataframe(scoring_log, run_dir / "scoring_log.pkl")

    holdout_predictions = results.get("holdout_predictions", pd.DataFrame())
    comparison = (
        build_holdout_comparison(holdout_predictions, checkpoint=checkpoint, horizon=horizon)
        if isinstance(holdout_predictions, pd.DataFrame) and "Obs_Window" in holdout_predictions
        else pd.DataFrame()
    )
    export_excel(
        report_path,
        comparison=comparison,
        forecasts=forecasts,
        regression=results.get("regression_evaluation", pd.DataFrame()),
        classification=results.get("classification_evaluation", pd.DataFrame()),
        calibration=results.get("classification_calibration", pd.DataFrame()),
        configuration_leaderboard=results.get("configuration_leaderboard"),
    )

    staged_dashboard = run_dir / "dashboard"
    staged_data_path = export_dashboard(report_path, staged_dashboard)
    dashboard_data = json.loads(staged_data_path.read_text(encoding="utf-8"))
    dashboard_path = _publish_dashboard(staged_dashboard, root / "artifacts" / "dashboard")

    return {
        "run_id": run_id,
        "sheet_name": sheet_name,
        "report_path": str(report_path),
        "dashboard_data_path": str(dashboard_path),
        "summary": dashboard_data["summary"],
    }
