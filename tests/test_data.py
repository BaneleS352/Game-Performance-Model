import pandas as pd
import pytest

from game_performance.data import CleaningConfig, clean_game_daily, load_raw_workbook


def _valid_rows(count=4):
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=count),
        "SystemName": [" SYS "] * count,
        "TournamentName": [" Game  Name "] * count,
        "Clients": [10] * count,
        "BetCount": [100] * count,
        "Stakes": [1000] * count,
        "GGR": [50] * count,
    })


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


def test_cleaner_reports_exact_and_game_date_duplicates_without_dropping_other_days():
    raw = _valid_rows(3)
    raw = pd.concat([raw, raw.iloc[[0]], raw.iloc[[1]].assign(Clients=11)], ignore_index=True)

    cleaned = clean_game_daily(raw)

    assert len(cleaned) == 3
    assert cleaned.attrs["completely_duplicated_rows_removed"] == 1
    assert cleaned.attrs["game_date_duplicates_removed"] == 1
    assert cleaned.attrs["duplicates_found"] == 2
    assert cleaned["TournamentName"].iloc[0] == "Game Name"


def test_cleaner_rejects_empty_or_fully_invalid_data_with_a_useful_report():
    with pytest.raises(ValueError, match="empty"):
        clean_game_daily(_valid_rows(0))

    raw = _valid_rows(1)
    raw["Clients"] = raw["Clients"].astype("object")
    raw.loc[0, "Clients"] = "not-a-number"
    cleaned = clean_game_daily(raw)
    assert cleaned.empty
    assert cleaned.attrs["invalid_rows"] == 1
    assert cleaned.attrs["missing_values_found"] == 0


def test_loader_rejects_an_empty_csv_with_a_clear_error(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV is empty"):
        load_raw_workbook(source)


def test_cleaner_flags_outliers_but_keeps_them_and_supports_configuration():
    raw = _valid_rows(5)
    raw["Stakes"] = [1000, 1050, 1100, 1150, 100_000]

    cleaned = clean_game_daily(raw)

    assert len(cleaned) == 5
    assert cleaned.attrs["outliers_detected"] == 1
    no_outlier_report = clean_game_daily(raw, config=CleaningConfig(detect_outliers=False))
    assert no_outlier_report.attrs["outliers_detected"] == 0


def test_cleaner_flags_missing_columns_and_invalid_numeric_values():
    raw = _valid_rows(2)
    raw["Clients"] = raw["Clients"].astype("object")
    raw.loc[0, "Clients"] = ""
    raw["Provider"] = [None, None]

    cleaned = clean_game_daily(raw, config=CleaningConfig(missing_column_threshold=0.5))

    assert len(cleaned) == 1
    assert cleaned.attrs["missing_values_found"] == 1
    assert cleaned.attrs["columns_with_excessive_missing_data"]["Provider"] == 1.0
    assert cleaned["Provider"].iloc[0] == "Unknown"
