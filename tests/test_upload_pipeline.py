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
            "BetCount": [1000] * 21,
            "Stakes": [100] * 21,
            "GGR": [20] * 21,
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


def test_save_uploaded_workbook_rejects_unsupported_files_and_sanitizes_name(tmp_path):
    with pytest.raises(ValueError, match="CSV or Excel"):
        upload_pipeline.save_uploaded_workbook(tmp_path, "scores.txt", b"data")

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
    assert Path(result["business_report_path"]).name == "Business Report.xlsx"
    assert Path(result["model_analysis_path"]).name == "Model Analysis.xlsx"
    assert Path(result["model_analysis_path"]).is_file()
    assert Path(result["dashboard_data_path"]).is_file()
    assert result["summary"]["games_scored"] == 1


def test_run_full_production_pipeline_runs_all_stages(tmp_path, monkeypatch):
    workbook = tmp_path / "source.xlsx"
    _raw_workbook(workbook)
    calls = []

    def build_features(cleaned, strategy):
        calls.append("features")
        return {(21, 7): cleaned.head(1)}

    def train_production(snapshots, settings, **kwargs):
        calls.append("training")
        assert kwargs["mode"] == "production"
        assert kwargs["max_families"] == 0
        assert kwargs["run_regression"] is True
        assert kwargs["run_classification"] is True
        return {
            "model_bundles": {(21, 7): {}},
            "holdout_predictions": pd.DataFrame(),
            "regression_evaluation": pd.DataFrame(),
            "classification_evaluation": pd.DataFrame(),
            "classification_calibration": pd.DataFrame(),
        }

    def score_production(*args, **kwargs):
        calls.append("scoring")
        return _forecasts(), pd.DataFrame()

    monkeypatch.setattr(upload_pipeline, "build_all_historical_snapshots", build_features)
    monkeypatch.setattr(upload_pipeline, "train_all_model_families", train_production)
    monkeypatch.setattr(upload_pipeline, "run_daily_lifecycle_scoring", score_production)

    progress = []
    result = upload_pipeline.run_full_production_pipeline(
        tmp_path,
        workbook,
        "Score me",
        progress_callback=lambda fraction, message: progress.append((fraction, message)),
    )

    assert calls == ["features", "training", "scoring"]
    assert [fraction for fraction, _ in progress] == [0.05, 0.18, 0.28, 0.76, 0.94, 1.0]
    assert Path(result["model_path"]).is_file()
    assert Path(result["business_report_path"]).is_file()
    assert Path(result["model_analysis_path"]).is_file()
    assert (tmp_path / "artifacts" / "processed" / "clean_game_daily.pkl").is_file()
    assert (tmp_path / "artifacts" / "features" / "historical_snapshots.joblib").is_file()
    assert result["training_metadata"]["pipeline_stages"] == [
        "01_data_exploration",
        "02_feature_engineering",
        "03_model_experiments",
        "04_training_and_forecasting",
    ]
    assert result["training_metadata"]["uploaded_data_days"] == 21
    assert result["training_metadata"]["temporal_split_fractions"] == {
        "training": 0.70,
        "validation": 0.15,
        "holdout": 0.15,
    }


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
