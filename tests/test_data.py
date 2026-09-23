import pandas as pd
import pytest

from game_performance.data import clean_game_daily


def test_formatted_currency_is_cleaned_before_numeric_coercion():
    raw = pd.DataFrame({
        "Date": ["2026-01-01"],
        "SystemName": ["SYS"],
        "TournamentName": ["Game"],
        "Clients": ["10"],
        "BetCount": ["1,234"],
        "Stakes": ["R1,234.50"],
        "GGR": ["R123.45"],
        "GGR_Perc": ["10%"],
    })
    cleaned = clean_game_daily(raw)
    assert cleaned.loc[0, "BetCount"] == 1234
    assert cleaned.loc[0, "Stakes"] == 1234.50
    assert cleaned.loc[0, "GGR"] == 123.45
    assert cleaned.loc[0, "Game_ID"] == "SYS::Game"


def test_bet_count_and_ggr_are_required_source_columns():
    raw = pd.DataFrame({
        "Date": ["2026-01-01"],
        "SystemName": ["SYS"],
        "TournamentName": ["Game"],
        "Clients": ["10"],
        "Stakes": ["R1,234.50"],
    })

    with pytest.raises(ValueError, match="BetCount"):
        clean_game_daily(raw)


def test_required_columns_are_case_insensitive_and_normalized():
    raw = pd.DataFrame({
        "date": ["2026-01-01"],
        "systemname": ["SYS"],
        "tournamentname": ["Game"],
        "clients": ["10"],
        "betcount": ["1,234"],
        "stakes": ["100"],
        "ggr": ["20"],
    })

    cleaned = clean_game_daily(raw)

    assert cleaned.loc[0, "BetCount"] == 1234


def test_cleaner_removes_corrupt_rows_and_duplicate_daily_records():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-01", "not-a-date", "2026-01-02", "2026-01-03", "2026-01-04"],
            "SystemName": ["SYS", "SYS", "SYS", "SYS", "SYS", ""],
            "TournamentName": ["Game", "Game", "Game", "Game", "Game", "Game"],
            "Clients": [10, 11, 10, "not-a-number", 10, 10],
            "BetCount": [100, 110, 100, 100, 100, 100],
            "Stakes": [100, 110, 100, 100, -5, 100],
            "GGR": [20, 22, 20, 20, 1, 20],
        }
    )

    cleaned = clean_game_daily(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "Clients"] == 11
    assert cleaned.attrs["input_rows"] == 6
    assert cleaned.attrs["corrupted_rows_removed"] == 4
    assert cleaned.attrs["duplicate_rows_removed"] == 1
    assert cleaned.attrs["output_rows"] == 1
