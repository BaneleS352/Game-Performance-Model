import pandas as pd

from game_performance.data import clean_game_daily
from game_performance.features import build_four_week_snapshot, build_lifecycle_snapshot


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


def test_four_week_snapshot_uses_week_4_targets():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=28),
        "SystemName": ["SYS"] * 28,
        "TournamentName": ["Game"] * 28,
        "Clients": [1] * 28,
        "Betcount": [2] * 28,
        "Stakes": [10] * 28,
        "GGR": [1] * 28,
    })
    data = clean_game_daily(raw)
    snapshot = build_four_week_snapshot(data)

    assert len(snapshot) == 1
    assert snapshot.loc[0, "Obs_Window"] == 14
    assert snapshot.loc[0, "Horizon"] == 14
    assert snapshot.loc[0, "actual_clients_week_3"] == 7
    assert snapshot.loc[0, "actual_clients_week_4"] == 7
    assert snapshot.loc[0, "target_clients"] == 7
    assert snapshot.loc[0, "target_stakes"] == 70


def test_four_week_snapshot_adds_week_1_and_2_training_signals():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=28),
        "SystemName": ["SYS"] * 28,
        "TournamentName": ["Game"] * 28,
        "Clients": ([10] * 7) + ([20] * 7) + ([30] * 14),
        "Betcount": [2] * 28,
        "Stakes": ([100] * 7) + ([150] * 7) + ([200] * 14),
        "GGR": [1] * 28,
    })
    data = clean_game_daily(raw)
    snapshot = build_four_week_snapshot(data)

    assert snapshot.loc[0, "clients_week_1_sum"] == 70
    assert snapshot.loc[0, "clients_week_2_sum"] == 140
    assert snapshot.loc[0, "clients_week_1_to_2_growth"] == 1
    assert snapshot.loc[0, "stakes_week_1_sum"] == 700
    assert snapshot.loc[0, "stakes_week_2_sum"] == 1050
    assert snapshot.loc[0, "stakes_week_1_to_2_growth"] == 0.5
    assert snapshot.loc[0, "Clients_Size_Band"] == "Medium"
    assert snapshot.loc[0, "Stakes_Size_Band"] == "Small"
