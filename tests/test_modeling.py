import numpy as np
import pandas as pd

import game_performance.modeling as modeling
from game_performance.config import Settings
from game_performance.modeling import (
    build_confidence_table,
    build_prediction_interval_table,
    confidence_from_prediction,
    inverse_transform_regression_target,
    prediction_interval_from_table,
    select_classification_threshold,
    select_best_configuration,
    transform_regression_target,
)


def test_select_best_configuration_uses_development_wape(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=8)
    snapshot = pd.DataFrame({
        "First_Observed_Date": dates.repeat(20),
        "target_clients": 1.0,
        "target_stakes": 1.0,
    })

    def fake_tune(development, target, checkpoint, horizon, categorical, settings, mode=None, training_window_days=None):
        score = 0.20 if (checkpoint, training_window_days) == (35, 60) else 0.80
        return {"_target_transform": "raw"}, pd.DataFrame({"Candidate": [1, 1], "WAPE": [score, score]})

    monkeypatch.setattr(modeling, "tune_regressor", fake_tune)
    settings = Settings(
        candidate_checkpoints=(21, 35),
        training_window_days=(None, 60),
        min_training_rows=10,
        min_validation_rows=2,
    )

    best, leaderboard = select_best_configuration(
        {(21, 7): snapshot, (35, 7): snapshot},
        settings,
    )

    assert (best["Checkpoint"], best["Training_Window_Days"]) == (35, 60)
    assert leaderboard.iloc[0]["Mean_WAPE"] == 0.20


def test_confidence_table_calibrates_prediction_ranges():
    table = build_confidence_table(
        actual=[100, 100, 1000, 1000],
        predicted=[95, 90, 500, 450],
        bins=2,
        min_bin_rows=1,
    )

    confidence = confidence_from_prediction([95, 500], table, [1, 1])

    assert confidence[0] > confidence[1]
    assert np.all((confidence >= 0) & (confidence <= 1))


def test_confidence_is_penalized_when_observation_coverage_is_low():
    table = {"fallback": 0.8, "bins": []}

    confidence = confidence_from_prediction([100, 100], table, [1.0, 0.5])

    assert confidence.tolist() == [0.8, 0.4]


def test_target_transforms_round_trip_business_scale_values():
    values = np.array([0.0, 10.0, 1000.0])

    transformed = transform_regression_target("target_clients", values)
    restored = inverse_transform_regression_target("target_clients", transformed)

    assert np.allclose(restored, values)
    assert np.all(restored >= 0)


def test_select_classification_threshold_uses_best_validation_f1():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.10, 0.40, 0.60, 0.90])

    threshold = select_classification_threshold(y_true, probability, default=0.50)
    predicted = (probability >= threshold).astype(int)

    assert threshold >= 0.50
    assert np.array_equal(predicted, y_true)


def test_prediction_ranges_keep_midpoint_inside_interval():
    table = build_prediction_interval_table(
        actual=[80, 100, 130],
        predicted=[100, 100, 100],
    )

    low, high = prediction_interval_from_table([100, 10], table)

    assert np.all(low <= np.array([100, 10]))
    assert np.all(high >= np.array([100, 10]))
    assert np.all(low >= 0)


def test_prediction_ranges_scale_with_validation_prediction_band():
    predicted = np.array([100.0] * 30 + [10_000.0] * 30)
    actual = np.array([95.0] * 30 + [8_000.0] * 30)
    table = build_prediction_interval_table(actual, predicted, bins=2, min_bin_rows=10)

    low, high = prediction_interval_from_table([100.0, 10_000.0], table)

    assert table["bins"]
    assert high[1] - low[1] > high[0] - low[0]


def test_prediction_ranges_keep_series_values_aligned_by_position():
    table = build_prediction_interval_table(
        actual=np.array([80.0, 120.0]),
        predicted=np.array([100.0, 100.0]),
    )

    assert table["lower_residual"] == -16.0
    assert table["upper_residual"] == 16.0
