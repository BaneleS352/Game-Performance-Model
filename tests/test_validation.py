import pandas as pd

from game_performance.validation import create_high_performance_target


def _frame(client_values, stake_values=None):
    n = len(client_values)
    stakes = [100.0] * n if stake_values is None else stake_values
    return pd.DataFrame({
        "target_clients": client_values,
        "target_stakes": stakes,
    })


def test_holdout_scoring_uses_development_reference_only():
    development = _frame([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    holdout = _frame([1000.0, 2000.0], [None, None])
    dev_scored, holdout_scored, thresholds = create_high_performance_target(development, holdout)

    assert dev_scored["future_performance_score"].notna().all()
    assert holdout_scored["future_performance_score"].notna().all()
    assert set(thresholds) == {"is_high_client_performer", "is_high_stakes_performer"}
    assert {"is_high_client_performer", "is_high_stakes_performer", "is_balanced_high_performer"}.issubset(
        dev_scored.columns
    )
    assert holdout_scored["is_high_performer"].equals(holdout_scored["is_balanced_high_performer"])
