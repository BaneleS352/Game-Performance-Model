import numpy as np
import pandas as pd

from game_performance.data import clean_game_daily
from game_performance.scoring import run_daily_lifecycle_scoring, score_snapshot


class DummyRegressor:
    def predict(self, frame):
        return np.array([np.log1p(24.0)] * len(frame))


class BandedRegressor:
    def predict(self, frame):
        return np.array([10.0, 24.0])


class DummyClassifier:
    def predict_proba(self, frame):
        return np.tile(np.array([[0.70, 0.30]]), (len(frame), 1))


class GoalClassifier:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile(np.array([[1 - self.probability, self.probability]]), (len(frame), 1))


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
        "confidence_tables": {"target_clients": {"fallback": 0.64, "bins": []}},
        "prediction_interval_tables": {"target_clients": {"lower_residual": -4.0, "upper_residual": 6.0}},
        "classifier": DummyClassifier(),
        "probability_calibrator": {"method": "test", "model": AdditiveCalibrator()},
        "classification_threshold": 0.65,
    }

    scored = score_snapshot(snapshot, bundle)

    assert np.isclose(scored.loc[0, "predicted_clients_week_4"], 24.0)
    assert np.isclose(scored.loc[0, "predicted_clients_week_4_low"], 20.0)
    assert np.isclose(scored.loc[0, "predicted_clients_week_4_high"], 30.0)
    assert np.isclose(scored.loc[0, "confidence_clients"], 0.64)
    assert np.isclose(scored.loc[0, "raw_probability_high_performance"], 0.30)
    assert np.isclose(scored.loc[0, "probability_high_performance"], 0.70)
    assert np.isclose(scored.loc[0, "classification_threshold"], 0.65)
    assert scored.loc[0, "predicted_high_performer"] == 1


def test_score_snapshot_emits_separate_client_and_stakes_goal_probabilities():
    snapshot = pd.DataFrame(
        {
            "Game_ID": ["SYS::Game"],
            "TournamentName": ["Game"],
            "SystemName": ["SYS"],
            "First_Observed_Date": [pd.Timestamp("2026-01-01")],
            "Lifecycle_Stage": ["Week 3"],
            "Obs_Window": [21],
            "Horizon": [7],
            "active_observation_days": [21],
            "observation_coverage": [1.0],
        }
    )
    bundle = {
        "categorical_features": [],
        "regressors": {},
        "classifiers": {
            "client": GoalClassifier(0.80),
            "stakes": GoalClassifier(0.40),
        },
        "probability_calibrators": {},
        "classification_thresholds": {"client": 0.65, "stakes": 0.65},
        "classification_threshold": 0.65,
    }

    scored = score_snapshot(snapshot, bundle)

    assert np.isclose(scored.loc[0, "probability_high_client_performer"], 0.80)
    assert np.isclose(scored.loc[0, "probability_high_stakes_performer"], 0.40)
    assert np.isclose(scored.loc[0, "probability_high_performance"], 0.40)
    assert scored.loc[0, "predicted_high_client_performer"] == 1
    assert scored.loc[0, "predicted_high_stakes_performer"] == 0
    assert scored.loc[0, "predicted_high_performer"] == 0


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


def test_daily_scoring_emits_all_eligible_checkpoint_families():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=28),
        "SystemName": ["SYS"] * 28,
        "TournamentName": ["Game"] * 28,
        "Clients": [10] * 28,
        "Stakes": [100] * 28,
    })
    data = clean_game_daily(raw)
    bundle = {
        "categorical_features": [],
        "regressors": {"target_clients": DummyRegressor()},
        "target_transforms": {"target_clients": "log"},
        "blend_weights": {"target_clients": 1.0},
        "confidence_tables": {"target_clients": {"fallback": 0.5, "bins": []}},
        "prediction_interval_tables": {"target_clients": {"lower_residual": -1, "upper_residual": 2}},
    }

    predictions, _ = run_daily_lifecycle_scoring(
        data,
        {(14, 14): bundle, (21, 7): bundle},
        {14: [14], 21: [7]},
    )

    assert set(predictions["Checkpoint"]) == {14, 21}
    assert {"predicted_clients_week_4_low", "predicted_clients_week_4_high"}.issubset(predictions.columns)
