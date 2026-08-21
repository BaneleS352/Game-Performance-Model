import pandas as pd

from game_performance.validation import create_high_performance_target


def _frame(ggr_values, missing_ggr_pct=None):
    n = len(ggr_values)
    ggr_pct = [0.1] * n if missing_ggr_pct is None else missing_ggr_pct
    return pd.DataFrame({
        "target_ggr": ggr_values,
        "target_stakes": [100.0] * n,
        "target_clients": [10.0] * n,
        "target_betcount": [20.0] * n,
        "target_ggr_pct": ggr_pct,
    })


def test_holdout_scoring_uses_development_reference_only():
    development = _frame([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])
    holdout = _frame([1000.0, 2000.0], [None, None])
    dev_scored, holdout_scored, _ = create_high_performance_target(development, holdout)
    assert dev_scored["future_performance_score"].notna().all()
    assert holdout_scored["future_performance_score"].notna().all()
