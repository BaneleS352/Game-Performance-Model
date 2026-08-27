import numpy as np

from game_performance.modeling import build_confidence_table, confidence_from_prediction


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
