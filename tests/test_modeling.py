import numpy as np
import pandas as pd

import game_performance.modeling as modeling
from game_performance.config import Settings
from game_performance.modeling import (
    build_conformal_interval_table,
    build_confidence_table,
    build_prediction_interval_table,
    confidence_from_prediction,
    derive_commercial_outputs,
    inverse_transform_regression_target,
    prediction_interval_from_table,
    select_classification_threshold,
    select_best_configuration,
    train_model_family,
    transform_regression_target,
)


def test_select_best_configuration_uses_development_wape(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=8)
    snapshot = pd.DataFrame({
        "First_Observed_Date": dates.repeat(20),
        "target_clients": 1.0,
        "target_stakes": 1.0,
    })

    validation_sets = []

    def fake_tune(
        development,
        target,
        checkpoint,
        horizon,
        categorical,
        settings,
        mode=None,
        training_window_days=None,
        validation=None,
    ):
        validation_sets.append(validation)
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
    assert all(validation is not None for validation in validation_sets)


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


def test_ggr_predictions_and_intervals_preserve_negative_values():
    values = np.array([-10.0, 0.0, 10.0])

    assert np.allclose(modeling.transform_regression_target("target_ggr", values), values)
    low, high = modeling.prediction_interval_from_table(
        [-5.0], {"lower_residual": -3.0, "upper_residual": 2.0}, non_negative=False
    )

    assert low.tolist() == [-8.0]
    assert high.tolist() == [-3.0]


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


def test_commercial_driver_predictions_recompose_business_outputs_exactly():
    outcomes = derive_commercial_outputs(
        {
            "target_clients": [2.0],
            "target_bets_per_client": [3.0],
            "target_stake_per_bet": [5.0],
            "target_ggr_margin": [-0.10],
        }
    )

    assert outcomes["target_clients"].tolist() == [2.0]
    assert outcomes["target_betcount"].tolist() == [6.0]
    assert outcomes["target_stakes"].tolist() == [30.0]
    assert outcomes["target_ggr"].tolist() == [-3.0]


def test_conformal_intervals_use_oof_absolute_residual_quantile():
    table = build_conformal_interval_table(
        actual=[10.0, 10.0, 20.0],
        predicted=[8.0, 12.0, 20.0],
        coverage=0.80,
    )
    low, high = prediction_interval_from_table([1.0, -1.0], table, non_negative=False)

    assert table == {
        "method": "split_conformal",
        "coverage": 0.80,
        "absolute_residual": 2.0,
        "rows": 3,
    }
    assert low.tolist() == [-1.0, -3.0]
    assert high.tolist() == [3.0, 1.0]


class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = value
        self.tree_count_ = 12

    def predict(self, frame):
        return np.full(len(frame), self.value)

    def get_best_iteration(self):
        return 11


def test_rolling_oof_folds_only_predict_later_dates(monkeypatch):
    frame = pd.DataFrame(
        {
            "First_Observed_Date": pd.date_range("2026-01-01", periods=8),
            "target_clients": np.arange(1, 9, dtype=float),
        }
    )
    observed_folds = []

    def fake_fit(model, train, validation, *args, **kwargs):
        observed_folds.append((train["First_Observed_Date"].max(), validation["First_Observed_Date"].min()))
        return model

    monkeypatch.setattr(modeling, "_build_regressor", lambda *args, **kwargs: _ConstantRegressor(1.0))
    monkeypatch.setattr(modeling, "_fit_regressor", fake_fit)
    oof, _ = modeling._rolling_oof_regression_predictions(
        frame,
        "target_clients",
        {"_target_transform": "raw"},
        [],
        Settings(rolling_validation_folds=2, min_training_rows=1, min_validation_rows=1),
        "quick",
        None,
    )

    assert not oof.empty
    assert observed_folds
    assert all(train_end < validation_start for train_end, validation_start in observed_folds)


def test_final_driver_refits_use_all_pre_holdout_rows(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=20)
    snapshot = pd.DataFrame(
        {
            "First_Observed_Date": dates,
            "SystemName": "System",
            "TournamentName": "Game",
            "Provider": "Provider A",
            "target_clients": 2.0,
            "target_bets_per_client": 3.0,
            "target_stake_per_bet": 4.0,
            "target_ggr_margin": 0.125,
            "target_betcount": 6.0,
            "target_stakes": 24.0,
            "target_ggr": 3.0,
        }
    )
    fitted = []

    def fake_oof(frame, target, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "actual": frame[target].to_numpy(),
                    "model_prediction": frame[target].to_numpy(),
                    "baseline_prediction": np.zeros(len(frame)),
                },
                index=frame.index,
            ),
            [12],
        )

    def fake_build(params, *args, **kwargs):
        return _ConstantRegressor(params["value"])

    def fake_fit(model, train, validation, target, *args, **kwargs):
        fitted.append((target, len(train), train["First_Observed_Date"].max(), validation))
        return model

    monkeypatch.setattr(modeling, "_rolling_oof_regression_predictions", fake_oof)
    monkeypatch.setattr(modeling, "_build_regressor", fake_build)
    monkeypatch.setattr(modeling, "_fit_regressor", fake_fit)
    settings = Settings(min_training_rows=1, min_validation_rows=1)
    shared_parameters = {
        "regressors": {
            "target_clients": {"value": 2.0, "_target_transform": "raw"},
            "target_bets_per_client": {"value": 3.0, "_target_transform": "raw"},
            "target_stake_per_bet": {"value": 4.0, "_target_transform": "raw"},
            "target_ggr_margin": {"value": 0.125, "_target_transform": "raw"},
        }
    }

    bundle, _, _, predictions, _ = train_model_family(
        snapshot,
        14,
        7,
        shared_parameters,
        14,
        7,
        settings,
        run_classification=False,
    )

    assert bundle["forecast_strategy"] == "commercial_drivers_v1"
    assert len(fitted) == 4
    assert all(rows == 17 and validation is None for _, rows, _, validation in fitted)
    assert all(train_end < pd.Timestamp("2026-01-18") for _, _, train_end, _ in fitted)
    assert predictions["predicted_stakes"].tolist() == [24.0, 24.0, 24.0]
    assert {"Age / Checkpoint", "Provider", "Volume"}.issubset(
        bundle["uncertainty_validation"]["Segment_Type"]
    )
    assert "Provider A" in bundle["uncertainty_validation"].loc[
        bundle["uncertainty_validation"]["Segment_Type"].eq("Provider"), "Segment"
    ].tolist()
