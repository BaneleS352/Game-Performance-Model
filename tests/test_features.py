import pandas as pd

from game_performance.data import clean_game_daily
from game_performance.features import build_lifecycle_snapshot


def test_live_snapshot_contains_no_future_targets():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=7),
        "SystemName": ["SYS"] * 7,
        "TournamentName": ["Game"] * 7,
        "Clients": range(1, 8),
        "Betcount": range(2, 9),
        "Stakes": range(10, 17),
        "GGR": range(1, 8),
    })
    data = clean_game_daily(raw)
    snapshot = build_lifecycle_snapshot(data, 7, 7, require_future=False)
    assert len(snapshot) == 1
    assert not any(column.startswith("target_") for column in snapshot.columns)
