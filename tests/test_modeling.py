import numpy as np

from game_performance.modeling import (
    build_confidence_table,
    confidence_from_prediction,
    inverse_transform_regression_target,
    select_classification_threshold,
    transform_regression_target,
)


def test_confidence_table_calibrates_prediction_ranges():
    table = build_confidence_table(
        actual=[100, 100, 1000, 1000],
        predicted=[95, 90, 500, 450],
        bins=2,
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
