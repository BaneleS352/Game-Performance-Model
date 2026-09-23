from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "Date",
    "SystemName",
    "TournamentName",
    "Clients",
    "BetCount",
    "Stakes",
    "GGR",
)


def _clean_numeric_text(series: pd.Series, *, percent: bool = False) -> pd.Series:
    """Parse formatted numeric text without destroying values before cleanup."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("R", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    if percent:
        # Source percentages may use decimal commas in some exports, for example 12,5%.
        cleaned = (
            cleaned
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
    else:
        # Assumes comma is used as a thousands separator, for example 1,234.56.
        cleaned = cleaned.str.replace(",", "", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def _clean_identifier(series: pd.Series) -> pd.Series:
    """Standardise identifier columns used to build Game_ID."""
    return series.astype("string").str.strip()


def load_raw_workbook(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Load the raw workbook from disk."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Source workbook not found: {path}. Set GAME_DATA_FILE to the correct path."
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the required source columns are present."""
    source_columns: dict[str, list[object]] = {}
    for column in df.columns:
        source_columns.setdefault(str(column).casefold(), []).append(column)

    missing = [column for column in REQUIRED_COLUMNS if column.casefold() not in source_columns]

    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    duplicates = [
        column for column in REQUIRED_COLUMNS
        if len(source_columns[column.casefold()]) > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate required source columns: {duplicates}")


def _canonicalize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match required upload headers case-insensitively and use canonical names."""
    validate_schema(df)
    source_columns = {str(column).casefold(): column for column in df.columns}
    return df.rename(columns={source_columns[column.casefold()]: column for column in REQUIRED_COLUMNS})


def clean_game_daily(
    df: pd.DataFrame,
    invalid_identifier_policy: str = "drop",
) -> pd.DataFrame:
    """Clean source rows and create stable lifecycle columns.

    Parameters
    ----------
    df:
        Raw daily game-performance data.

    invalid_identifier_policy:
        "drop" removes rows where SystemName or TournamentName is missing or blank.

        "raise" raises a ValueError instead.
    """
    if invalid_identifier_policy not in {"drop", "raise"}:
        raise ValueError("invalid_identifier_policy must be 'drop' or 'raise'.")

    result = _canonicalize_required_columns(df.copy())

    input_rows = len(result)

    # Dates
    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    invalid_dates = result["Date"].isna()

    # Numeric fields
    for column in ("Clients", "BetCount", "Stakes", "GGR"):
        if column in result.columns:
            result[column] = _clean_numeric_text(result[column])

    if "GGR_Perc" in result.columns:
        result["GGR_Perc"] = _clean_numeric_text(
            result["GGR_Perc"],
            percent=True,
        )

    # Game identifiers
    system = _clean_identifier(result["SystemName"])
    tournament = _clean_identifier(result["TournamentName"])

    invalid_identifier = (
        system.isna()
        | tournament.isna()
        | system.eq("")
        | tournament.eq("")
    )

    invalid_identifier_count = int(invalid_identifier.sum())

    if invalid_identifier_count > 0:
        if invalid_identifier_policy == "raise":
            raise ValueError(
                "Cannot construct Game_ID for "
                f"{invalid_identifier_count} rows with missing SystemName/TournamentName."
            )

    required_numeric = ("Clients", "BetCount", "Stakes", "GGR")
    invalid_numeric = result[list(required_numeric)].isna().any(axis=1)
    invalid_numeric |= ~np.isfinite(result[list(required_numeric)]).all(axis=1)
    negative_activity = result[["Clients", "BetCount", "Stakes"]].lt(0).any(axis=1)
    corrupted = invalid_dates | invalid_identifier | invalid_numeric | negative_activity

    # A daily game record must have one valid row per game and date. Keep the
    # last valid source occurrence when an export repeats a daily record.
    valid = result.loc[~corrupted].copy()
    valid["SystemName"] = _clean_identifier(valid["SystemName"])
    valid["TournamentName"] = _clean_identifier(valid["TournamentName"])
    valid["Game_ID"] = valid["SystemName"] + "::" + valid["TournamentName"]
    duplicate_mask = valid.duplicated(subset=["Date", "Game_ID"], keep="last")
    duplicate_count = int(duplicate_mask.sum())
    result = valid.loc[~duplicate_mask].copy().reset_index(drop=True)
    system = result["SystemName"]
    tournament = result["TournamentName"]

    # Store auditable counts for the notebook/reporting layer.
    result.attrs.update(
        {
            "input_rows": int(input_rows),
            "invalid_date_rows": int(invalid_dates.sum()),
            "excluded_invalid_identifier_rows": invalid_identifier_count,
            "invalid_numeric_rows": int(invalid_numeric.sum()),
            "negative_activity_rows": int(negative_activity.sum()),
            "corrupted_rows_removed": int(corrupted.sum()),
            "duplicate_rows_removed": duplicate_count,
            "output_rows": int(len(result)),
        }
    )

    # Lifecycle reference fields
    result["First_Observed_Date"] = (
        result.groupby("Game_ID")["Date"]
        .transform("min")
    )

    result["Days_Since_First_Observed"] = (
        result["Date"] - result["First_Observed_Date"]
    ).dt.days + 1

    # Recalculated GGR percentage
    if {"GGR", "Stakes"}.issubset(result.columns):
        result["GGR_Perc_Calc"] = np.where(
            result["Stakes"].ne(0),
            result["GGR"] / result["Stakes"],
            np.nan,
        )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def data_quality_summary(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame | tuple[int, int]]:
    """Return simple quality checks for the cleaned dataset."""
    return {
        "shape": df.shape,
        "missing": (
            df.isna()
            .sum()
            .sort_values(ascending=False)
            .to_frame("missing_count")
        ),
        "numeric_summary": df.select_dtypes(include=["number"]).describe().T,
        "date_range": pd.DataFrame(
            {
                "min_date": [df["Date"].min()],
                "max_date": [df["Date"].max()],
                "games": [df["Game_ID"].nunique()],
            }
        ),
        "attrs": pd.DataFrame(
            {
                "metric": list(df.attrs.keys()),
                "value": list(df.attrs.values()),
            }
        ),
    }
