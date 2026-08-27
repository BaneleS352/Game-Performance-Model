from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORECAST_FIELDS = [
    "Game_ID",
    "TournamentName",
    "SystemName",
    "First_Observed_Date",
    "predicted_clients_week_4",
    "confidence_clients",
    "predicted_stakes_week_4",
    "confidence_stakes",
    "active_observation_days",
    "observation_coverage",
    "As_Of_Date",
]


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


def _safe_rate(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean()) if not series.empty else 0.0


def build_dashboard_data(workbook_path: str | Path) -> dict[str, Any]:
    workbook_path = Path(workbook_path)
    forecasts = pd.read_excel(workbook_path, sheet_name="Forecasts")
    holdout = pd.read_excel(workbook_path, sheet_name="Holdout Comparison")

    forecasts["predicted_clients_week_4"] = pd.to_numeric(
        forecasts["predicted_clients_week_4"],
        errors="coerce",
    ).fillna(0)
    forecasts["predicted_stakes_week_4"] = pd.to_numeric(
        forecasts["predicted_stakes_week_4"],
        errors="coerce",
    ).fillna(0)

    system_summary = (
        forecasts.groupby("SystemName", dropna=False)
        .agg(
            games=("Game_ID", "count"),
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
        "confidence_clients": _safe_rate(forecasts["confidence_clients"]),
        "confidence_stakes": _safe_rate(forecasts["confidence_stakes"]),
        "holdout_off_by_10_rate": _safe_rate(holdout.get("off_by_10_pct", pd.Series(dtype=float))),
        "holdout_off_by_20_rate": _safe_rate(holdout.get("off_by_20_pct", pd.Series(dtype=float))),
        "avg_probability_clients": _safe_rate(holdout.get("probability_clients", pd.Series(dtype=float))),
        "avg_probability_stakes": _safe_rate(holdout.get("probability_stakes", pd.Series(dtype=float))),
        "as_of_date": _clean_value(forecasts["As_Of_Date"].max()) if "As_Of_Date" in forecasts else None,
    }

    top_stakes = forecasts.sort_values("predicted_stakes_week_4", ascending=False).head(250)
    top_clients = forecasts.sort_values("predicted_clients_week_4", ascending=False).head(250)
    review_candidates = forecasts.assign(
        value_rank=forecasts["predicted_stakes_week_4"].rank(ascending=False, method="first"),
        coverage_gap=1 - pd.to_numeric(forecasts.get("observation_coverage", 1), errors="coerce").fillna(1),
    ).sort_values(["coverage_gap", "predicted_stakes_week_4"], ascending=[False, False]).head(250)

    return {
        "summary": summary,
        "systems": _records(
            system_summary,
            [
                "SystemName",
                "games",
                "predicted_clients_week_4",
                "predicted_stakes_week_4",
                "confidence_clients",
                "confidence_stakes",
            ],
        ),
        "forecasts": _records(forecasts, FORECAST_FIELDS),
        "top_stakes": _records(top_stakes, FORECAST_FIELDS),
        "top_clients": _records(top_clients, FORECAST_FIELDS),
        "review_candidates": _records(review_candidates, FORECAST_FIELDS),
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
