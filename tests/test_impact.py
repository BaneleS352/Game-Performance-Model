import pandas as pd

from game_performance.impact import add_external_impact_flags, detect_external_impact_periods


def _daily(values):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=len(values)),
            "Game_ID": "SYS::Game",
            "SystemName": "SYS",
            "TournamentName": "Game",
            "Clients": values,
            "BetCount": 100,
            "Stakes": 100,
            "GGR": 20,
        }
    )


def test_detects_spike_and_drop():
    periods = detect_external_impact_periods(_daily([10, 10, 10, 20, 5]))

    assert set(periods["Direction"]) == {"Spike", "Drop"}
    assert set(periods["Metric"]) == {"Clients"}


def test_ignores_stable_data_and_groups_adjacent_flags():
    periods = detect_external_impact_periods(_daily([10, 10, 10, 20, 20, 20, 5]))

    assert len(periods) == 2
    spike = periods[periods["Direction"] == "Spike"].iloc[0]
    assert spike["Start_Date"] == pd.Timestamp("2026-01-04")
    assert spike["End_Date"] == pd.Timestamp("2026-01-06")


def test_adds_recent_impact_flags_to_forecasts():
    periods = detect_external_impact_periods(_daily([10, 10, 10, 20]))
    forecasts = add_external_impact_flags(
        pd.DataFrame({"Game_ID": ["SYS::Game"], "As_Of_Date": [pd.Timestamp("2026-01-04")]}),
        periods,
    )

    assert bool(forecasts.loc[0, "external_impact_detected"])
    assert "2026-01-04" in forecasts.loc[0, "external_impact_periods"]
