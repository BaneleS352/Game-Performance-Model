import pandas as pd

from game_performance.data import clean_game_daily
from game_performance.features import build_lifecycle_snapshot, prepare_features


def _daily_data(periods=28, **metrics):
    return clean_game_daily(pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=periods),
        "SystemName": ["SYS"] * periods,
        "TournamentName": ["Game"] * periods,
        "Clients": metrics.get("Clients", [1] * periods),
        "Betcount": metrics.get("Betcount", [2] * periods),
        "Stakes": metrics.get("Stakes", [10] * periods),
        "GGR": metrics.get("GGR", [1] * periods),
    }))


def test_live_snapshot_contains_no_future_targets():
    snapshot = build_lifecycle_snapshot(
        _daily_data(7, Clients=range(1, 8), Betcount=range(20, 27), Stakes=range(10, 17), GGR=range(2, 9)),
        7,
        7,
        require_future=False,
    )

    assert len(snapshot) == 1
    assert not any(column.startswith("target_") for column in snapshot.columns)
    assert snapshot.loc[0, "clients_first"] == 1
    assert snapshot.loc[0, "clients_last"] == 7
    assert snapshot.loc[0, "clients_positive_days"] == 7
    assert snapshot.loc[0, "Observation_End_Weekday"] == "Wednesday"


def test_historical_snapshot_targets_all_regression_metrics():
    snapshot = build_lifecycle_snapshot(
        _daily_data(14, Clients=range(1, 15), Betcount=range(20, 34), Stakes=range(10, 24), GGR=range(2, 16)),
        7,
        7,
    )

    assert [column for column in snapshot.columns if column.startswith("target_")] == [
        "target_clients", "target_stakes", "target_betcount", "target_ggr",
        "target_clients_week_1", "target_clients_week_2",
        "target_stakes_week_1", "target_stakes_week_2",
    ]


def test_snapshot_targets_the_next_seven_days():
    snapshot = build_lifecycle_snapshot(_daily_data(), 14, 7)

    assert len(snapshot) == 1
    assert snapshot.loc[0, "Obs_Window"] == 14
    assert snapshot.loc[0, "Horizon"] == 7
    assert snapshot.loc[0, "Prediction_Start"] == pd.Timestamp("2026-01-15")
    assert snapshot.loc[0, "Prediction_End"] == pd.Timestamp("2026-01-21")
    assert snapshot.loc[0, "target_clients"] == 7
    assert snapshot.loc[0, "target_stakes"] == 70
    assert snapshot.loc[0, "target_betcount"] == 14
    assert snapshot.loc[0, "target_ggr"] == 7


def test_latest_snapshot_uses_the_final_as_of_date_for_the_forecast_window():
    data = _daily_data(35, Clients=([1] * 14) + ([10] * 21))
    as_of_date = data["Date"].max()

    snapshot = build_lifecycle_snapshot(data, 21, 7, require_future=False, as_of_date=as_of_date)

    assert snapshot.loc[0, "Lifecycle_Stage"] == "Latest 21-day window"
    assert snapshot.loc[0, "Prediction_Start"] == as_of_date + pd.Timedelta(days=1)
    assert snapshot.loc[0, "Prediction_End"] == as_of_date + pd.Timedelta(days=7)
    assert snapshot.loc[0, "clients_week_1_sum"] == 70
    assert snapshot.loc[0, "clients_week_3_sum"] == 70


def test_weekly_features_reach_model_features():
    snapshot = build_lifecycle_snapshot(
        _daily_data(
            Clients=([10] * 7) + ([20] * 7) + ([20] * 14),
            Stakes=([100] * 7) + ([400] * 7) + ([300] * 14),
            Betcount=([20] * 7) + ([40] * 7) + ([30] * 7) + ([25] * 7),
            GGR=([10] * 7) + ([40] * 7) + ([30] * 14),
        ),
        21,
        7,
    )
    features = prepare_features(snapshot)

    assert snapshot.loc[0, "betcount_week_2_to_3_growth"] == -0.25
    assert snapshot.loc[0, "stake_per_bet_week_2_to_3_growth"] == 0
    assert features.loc[0, "betcount_slope"] > 0
    assert "betcount_recent_share" in features.columns
    assert "log_clients_week_3_sum" in features.columns
