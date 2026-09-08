from pathlib import Path

import pandas as pd
import pytest

from game_performance import upload_pipeline


def _raw_workbook(path: Path, *, sheet_name: str = "Score me") -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=21),
            "SystemName": ["SYS"] * 21,
            "TournamentName": ["Game"] * 21,
            "Clients": [10] * 21,
            "Stakes": [100] * 21,
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Game_ID": ["SYS::Game"],
            "TournamentName": ["Game"],
            "SystemName": ["SYS"],
            "predicted_clients": [50],
            "predicted_stakes": [1000],
            "confidence_clients": [0.8],
            "confidence_stakes": [0.7],
            "Checkpoint": [21],
            "As_Of_Date": [pd.Timestamp("2026-02-01")],
        }
    )


def test_save_uploaded_workbook_rejects_non_xlsx_and_sanitizes_name(tmp_path):
    with pytest.raises(ValueError, match=".xlsx"):
        upload_pipeline.save_uploaded_workbook(tmp_path, "scores.csv", b"data")

    source = tmp_path / "source.xlsx"
    _raw_workbook(source)
    upload_id, saved = upload_pipeline.save_uploaded_workbook(
        tmp_path,
        "../Quarterly scores!.xlsx",
        source.read_bytes(),
    )

    assert upload_id in saved.name
    assert saved.name.endswith("Quarterly_scores.xlsx")
    assert saved.parent == tmp_path / "data" / "uploads"


def test_score_uploaded_workbook_uses_selected_sheet(tmp_path, monkeypatch):
    workbook = tmp_path / "source.xlsx"
    _raw_workbook(workbook, sheet_name="Selected")
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    loaded_sheets = []

    original_loader = upload_pipeline.load_raw_workbook

    def load_selected(path, sheet_name=0):
        loaded_sheets.append(sheet_name)
        return original_loader(path, sheet_name=sheet_name)

    monkeypatch.setattr(upload_pipeline, "load_raw_workbook", load_selected)
    monkeypatch.setattr(upload_pipeline, "load_artifact", lambda _: {
        "model_bundles": {(21, 7): {}},
        "selection_evaluation": {"Checkpoint": 21, "Horizon": 7},
    })
    monkeypatch.setattr(upload_pipeline, "run_daily_lifecycle_scoring", lambda *args, **kwargs: (_forecasts(), pd.DataFrame()))

    result = upload_pipeline.score_uploaded_workbook(tmp_path, workbook, "Selected", model_path=model_path)

    assert loaded_sheets == ["Selected"]
    assert Path(result["report_path"]).is_file()
    assert Path(result["dashboard_data_path"]).is_file()
    assert result["summary"]["games_scored"] == 1


def test_score_uploaded_workbook_reports_missing_model_and_ineligible_games(tmp_path, monkeypatch):
    workbook = tmp_path / "source.xlsx"
    _raw_workbook(workbook)

    with pytest.raises(FileNotFoundError, match="Trained model artifact"):
        upload_pipeline.score_uploaded_workbook(tmp_path, workbook, "Score me")

    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(upload_pipeline, "load_artifact", lambda _: {
        "model_bundles": {(21, 7): {}},
        "selection_evaluation": {"Checkpoint": 21, "Horizon": 7},
    })
    monkeypatch.setattr(upload_pipeline, "run_daily_lifecycle_scoring", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))

    with pytest.raises(ValueError, match="No games are eligible"):
        upload_pipeline.score_uploaded_workbook(tmp_path, workbook, "Score me", model_path=model_path)


def test_score_uploaded_workbook_rejects_missing_required_columns(tmp_path):
    workbook = tmp_path / "invalid.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"Date": ["2026-01-01"]}).to_excel(writer, sheet_name="Invalid", index=False)

    with pytest.raises(ValueError, match="Missing required source columns"):
        upload_pipeline.score_uploaded_workbook(tmp_path, workbook, "Invalid", model_path=tmp_path / "missing.joblib")
