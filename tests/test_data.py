import pandas as pd
import pytest

from game_performance.data import clean_game_daily


def test_formatted_currency_is_cleaned_before_numeric_coercion():
    raw = pd.DataFrame({
        "Date": ["2026-01-01"],
        "SystemName": ["SYS"],
        "TournamentName": ["Game"],
        "Clients": ["10"],
        "Betcount": ["1,234"],
        "Stakes": ["R1,234.50"],
        "GGR": ["R123.45"],
        "GGR_Perc": ["10%"],
    })
    cleaned = clean_game_daily(raw)
    assert cleaned.loc[0, "Betcount"] == 1234
    assert cleaned.loc[0, "Stakes"] == 1234.50
    assert cleaned.loc[0, "GGR"] == 123.45
    assert cleaned.loc[0, "Game_ID"] == "SYS::Game"


def test_betcount_and_ggr_are_required_source_columns():
    raw = pd.DataFrame({
        "Date": ["2026-01-01"],
        "SystemName": ["SYS"],
        "TournamentName": ["Game"],
        "Clients": ["10"],
        "Stakes": ["R1,234.50"],
    })

    with pytest.raises(ValueError, match="Betcount"):
        clean_game_daily(raw)
