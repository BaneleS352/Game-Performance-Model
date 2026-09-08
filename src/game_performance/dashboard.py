from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FORECAST_COLUMNS = {
    "Game_ID",
    "TournamentName",
    "SystemName",
    "Lifecycle_Stage",
    "Obs_Window",
    "Horizon",
    "First_Observed_Date",
    "predicted_clients_week_4_low",
    "predicted_clients_week_4",
    "predicted_clients_week_4_high",
    "confidence_clients",
    "predicted_stakes_week_4_low",
    "predicted_stakes_week_4",
    "predicted_stakes_week_4_high",
    "confidence_stakes",
    "probability_high_client_performer",
    "predicted_high_client_performer",
    "probability_high_stakes_performer",
    "predicted_high_stakes_performer",
    "raw_probability_high_performance",
    "probability_high_performance",
    "classification_threshold",
    "predicted_high_performer",
    "active_observation_days",
    "observation_coverage",
    "Checkpoint",
    "As_Of_Date",
    "scored_at",
}
DEFAULT_SHEET_PREVIEW_ROWS = 250


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _records(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    available = [field for field in fields if field in frame.columns]
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in frame[available].replace([np.inf, -np.inf], np.nan).to_dict("records")
    ]


def _label(column: str) -> str:
    return column.replace("_", " ").replace("pct", "%").title()


def _column_type(column: str, series: pd.Series) -> str:
    lowered = column.lower()
    if "date" in lowered or lowered.endswith("_at"):
        return "date"
    if any(token in lowered for token in ("confidence", "probability", "coverage", "rate", "pct")):
        return "percent"
    if "stakes" in lowered or "ggr" in lowered:
        return "money"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


def _column_metadata(frame: pd.DataFrame, default_columns: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "key": column,
            "label": _label(column),
            "type": _column_type(column, frame[column]),
            "defaultVisible": column in default_columns,
        }
        for column in frame.columns
    ]


def _workbook_sheet_previews(workbook_path: Path, *, limit: int = DEFAULT_SHEET_PREVIEW_ROWS) -> list[dict[str, Any]]:
    workbook = pd.ExcelFile(workbook_path)
    previews = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
        fields = list(frame.columns)
        previews.append(
            {
                "name": sheet_name,
                "totalRows": int(len(frame)),
                "columns": _column_metadata(frame, set(fields[:8])),
                "rows": _records(frame.head(limit), fields),
            }
        )
    return previews


def _safe_rate(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean()) if not series.empty else 0.0


def _latest_checkpoint(frame: pd.DataFrame, checkpoint_column: str) -> pd.DataFrame:
    if checkpoint_column not in frame.columns:
        return frame
    return (
        frame.assign(_checkpoint=pd.to_numeric(frame[checkpoint_column], errors="coerce").fillna(-1))
        .sort_values(["Game_ID", "_checkpoint"], kind="stable")
        .drop_duplicates("Game_ID", keep="last")
        .drop(columns="_checkpoint")
    )


def build_dashboard_data(workbook_path: str | Path) -> dict[str, Any]:
    workbook_path = Path(workbook_path)
    forecasts = _latest_checkpoint(pd.read_excel(workbook_path, sheet_name="Forecasts"), "Checkpoint")
    holdout = _latest_checkpoint(pd.read_excel(workbook_path, sheet_name="Holdout Comparison"), "Obs_Window")
    forecast_fields = list(forecasts.columns)

    for metric in ("clients", "stakes"):
        generic = f"predicted_{metric}"
        legacy = f"predicted_{metric}_week_4"
        if generic not in forecasts and legacy in forecasts:
            forecasts[generic] = forecasts[legacy]
        for suffix in ("_low", "_high"):
            generic_band = f"predicted_{metric}{suffix}"
            legacy_band = f"predicted_{metric}_week_4{suffix}"
            if generic_band not in forecasts and legacy_band in forecasts:
                forecasts[generic_band] = forecasts[legacy_band]

    forecasts["predicted_clients"] = pd.to_numeric(
        forecasts["predicted_clients"],
        errors="coerce",
    ).fillna(0)
    forecasts["predicted_stakes"] = pd.to_numeric(
        forecasts["predicted_stakes"],
        errors="coerce",
    ).fillna(0)
    forecasts["predicted_clients_week_4"] = forecasts["predicted_clients"]
    forecasts["predicted_stakes_week_4"] = forecasts["predicted_stakes"]

    system_summary = (
        forecasts.groupby("SystemName", dropna=False)
        .agg(
            games=("Game_ID", "count"),
            predicted_clients=("predicted_clients", "sum"),
            predicted_stakes=("predicted_stakes", "sum"),
            predicted_clients_week_4=("predicted_clients_week_4", "sum"),
            predicted_stakes_week_4=("predicted_stakes_week_4", "sum"),
            confidence_clients=("confidence_clients", "mean"),
            confidence_stakes=("confidence_stakes", "mean"),
        )
        .reset_index()
        .sort_values("predicted_stakes_week_4", ascending=False)
    )

    summary = {
        "source_workbook": str(workbook_path),
        "games_scored": int(len(forecasts)),
        "systems_scored": int(forecasts["SystemName"].nunique()),
        "holdout_games": int(len(holdout)),
            "total_predicted_clients_week_4": float(forecasts["predicted_clients_week_4"].sum()),
            "total_predicted_stakes_week_4": float(forecasts["predicted_stakes_week_4"].sum()),
            "total_predicted_clients": float(forecasts["predicted_clients"].sum()),
            "total_predicted_stakes": float(forecasts["predicted_stakes"].sum()),
        "confidence_clients": _safe_rate(forecasts["confidence_clients"]),
        "confidence_stakes": _safe_rate(forecasts["confidence_stakes"]),
        "holdout_off_by_10_rate": _safe_rate(holdout.get("off_by_10_pct", pd.Series(dtype=float))),
        "holdout_off_by_20_rate": _safe_rate(holdout.get("off_by_20_pct", pd.Series(dtype=float))),
        "avg_probability_clients": _safe_rate(holdout.get("probability_clients", pd.Series(dtype=float))),
        "avg_probability_stakes": _safe_rate(holdout.get("probability_stakes", pd.Series(dtype=float))),
        "as_of_date": _clean_value(forecasts["As_Of_Date"].max()) if "As_Of_Date" in forecasts else None,
    }

    top_stakes = forecasts.sort_values("predicted_stakes", ascending=False).head(250)
    top_clients = forecasts.sort_values("predicted_clients", ascending=False).head(250)
    observation_coverage = pd.to_numeric(
        forecasts.get("observation_coverage", pd.Series(1.0, index=forecasts.index)),
        errors="coerce",
    ).fillna(1)
    review_candidates = forecasts.assign(
        value_rank=forecasts["predicted_stakes"].rank(ascending=False, method="first"),
        coverage_gap=1 - observation_coverage,
    ).sort_values(["coverage_gap", "predicted_stakes"], ascending=[False, False]).head(250)

    return {
        "summary": summary,
        "systems": _records(
            system_summary,
            [
                "SystemName",
                "games",
                "predicted_clients",
                "predicted_stakes",
                "predicted_clients_week_4",
                "predicted_stakes_week_4",
                "confidence_clients",
                "confidence_stakes",
            ],
        ),
        "columns": {
            "forecasts": _column_metadata(forecasts[forecast_fields], DEFAULT_FORECAST_COLUMNS),
        },
        "forecasts": _records(forecasts, forecast_fields),
        "top_stakes": _records(top_stakes, forecast_fields),
        "top_clients": _records(top_clients, forecast_fields),
        "review_candidates": _records(review_candidates, forecast_fields),
        "workbook_sheets": _workbook_sheet_previews(workbook_path),
    }


def export_dashboard(
    workbook_path: str | Path = "artifacts/reports/model_business_report.xlsx",
    output_dir: str | Path = "artifacts/dashboard",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_dashboard_data(workbook_path)
    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    json_path = output_dir / "dashboard_data.json"
    js_path = output_dir / "dashboard_data.js"
    json_path.write_text(json_text, encoding="utf-8")
    js_path.write_text(f"window.DASHBOARD_DATA = {json_text};\n", encoding="utf-8")
    return json_path


if __name__ == "__main__":
    print(export_dashboard())
