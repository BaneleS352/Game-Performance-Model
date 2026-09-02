import numpy as np
import pandas as pd

from game_performance.scoring import score_snapshot


class DummyRegressor:
    def predict(self, frame):
        return np.array([np.log1p(24.0)] * len(frame))


class BandedRegressor:
    def predict(self, frame):
        return np.array([10.0, 24.0])


class DummyClassifier:
    def predict_proba(self, frame):
        return np.tile(np.array([[0.70, 0.30]]), (len(frame), 1))


class AdditiveCalibrator:
    def predict(self, probability):
        return np.asarray(probability) + 0.40


def test_score_snapshot_uses_calibrated_probability_and_bundle_threshold():
    snapshot = pd.DataFrame(
        {
            "Game_ID": ["SYS::Game"],
            "TournamentName": ["Game"],
            "SystemName": ["SYS"],
            "First_Observed_Date": [pd.Timestamp("2026-01-01")],
            "Lifecycle_Stage": ["Day 7"],
            "Obs_Window": [7],
            "Horizon": [7],
            "active_observation_days": [7],
            "observation_coverage": [1.0],
        }
    )
    bundle = {
        "categorical_features": [],
        "regressors": {"target_clients": DummyRegressor()},
        "target_transforms": {"target_clients": "log"},
        "blend_weights": {"target_clients": 1.0},
        "confidence_scores": {"target_clients": 0.64},
        "confidence_tables": {"target_clients": {"fallback": 1.0, "bins": []}},
        "classifier": DummyClassifier(),
        "probability_calibrator": {"method": "test", "model": AdditiveCalibrator()},
        "classification_threshold": 0.65,
    }

    scored = score_snapshot(snapshot, bundle)

    assert np.isclose(scored.loc[0, "predicted_clients_week_4"], 24.0)
    assert np.isclose(scored.loc[0, "confidence_clients"], 0.64)
    assert np.isclose(scored.loc[0, "raw_probability_high_performance"], 0.30)
    assert np.isclose(scored.loc[0, "probability_high_performance"], 0.70)
    assert np.isclose(scored.loc[0, "classification_threshold"], 0.65)
    assert scored.loc[0, "predicted_high_performer"] == 1


def test_score_snapshot_uses_prediction_bands_for_game_confidence():
    snapshot = pd.DataFrame(
        {
            "Game_ID": ["SYS::Game A", "SYS::Game B"],
            "TournamentName": ["Game A", "Game B"],
            "SystemName": ["SYS", "SYS"],
            "First_Observed_Date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "Lifecycle_Stage": ["New", "New"],
            "Obs_Window": [14, 14],
            "Horizon": [14, 14],
            "active_observation_days": [14, 14],
            "observation_coverage": [1.0, 1.0],
        }
    )
    bundle = {
        "categorical_features": [],
        "regressors": {"target_clients": BandedRegressor()},
        "target_transforms": {"target_clients": "raw"},
        "blend_weights": {"target_clients": 1.0},
        "confidence_scores": {"target_clients": 0.64},
        "confidence_tables": {
            "target_clients": {
                "fallback": 0.64,
                "bins": [
                    {"min_pred": 0, "max_pred": 15, "confidence": 0.80},
                    {"min_pred": 15.01, "max_pred": 100, "confidence": 0.35},
                ],
            }
        },
    }

    scored = score_snapshot(snapshot, bundle)

    assert scored["confidence_clients"].tolist() == [0.8, 0.35]
