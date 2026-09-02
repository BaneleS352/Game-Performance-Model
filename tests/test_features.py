import pandas as pd

from game_performance.data import clean_game_daily
from game_performance.features import build_four_week_snapshot, build_lifecycle_snapshot


def test_live_snapshot_contains_no_future_targets():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=7),
        "SystemName": ["SYS"] * 7,
        "TournamentName": ["Game"] * 7,
        "Clients": range(1, 8),
        "Stakes": range(10, 17),
    })
    data = clean_game_daily(raw)
    snapshot = build_lifecycle_snapshot(data, 7, 7, require_future=False)

    assert len(snapshot) == 1
    assert not any(column.startswith("target_") for column in snapshot.columns)
    assert snapshot.loc[0, "clients_first"] == 1
    assert snapshot.loc[0, "clients_last"] == 7
    assert snapshot.loc[0, "clients_positive_days"] == 7
    assert snapshot.loc[0, "Observation_End_Weekday"] == "Wednesday"


def test_historical_snapshot_targets_clients_and_stakes_only():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=14),
        "SystemName": ["SYS"] * 14,
        "TournamentName": ["Game"] * 14,
        "Clients": range(1, 15),
        "Stakes": range(10, 24),
    })
    data = clean_game_daily(raw)
    snapshot = build_lifecycle_snapshot(data, 7, 7, require_future=True)

    target_columns = [column for column in snapshot.columns if column.startswith("target_")]
    assert target_columns == [
        "target_clients",
        "target_stakes",
        "target_clients_week_1",
        "target_clients_week_2",
        "target_stakes_week_1",
        "target_stakes_week_2",
    ]


def test_four_week_snapshot_uses_week_4_targets():
    raw = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=28),
        "SystemName": ["SYS"] * 28,
        "TournamentName": ["Game"] * 28,
        "Clients": [1] * 28,
        "Stakes": [10] * 28,
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
        "Stakes": ([100] * 7) + ([150] * 7) + ([200] * 14),
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
