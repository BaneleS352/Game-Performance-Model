from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "SystemName",
    "TournamentName",
    "Clients",
    "Betcount",
    "Stakes",
    "GGR",
}


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

    return pd.read_excel(path, sheet_name=sheet_name)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the required source columns are present."""
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")


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
    validate_schema(df)

    if invalid_identifier_policy not in {"drop", "raise"}:
        raise ValueError("invalid_identifier_policy must be 'drop' or 'raise'.")

    result = df.copy()

    # Dates
    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    invalid_dates = int(result["Date"].isna().sum())
    result.attrs["invalid_date_rows"] = invalid_dates

    # Numeric fields
    result["Clients"] = _clean_numeric_text(result["Clients"])
    result["Betcount"] = _clean_numeric_text(result["Betcount"])
    result["Stakes"] = _clean_numeric_text(result["Stakes"])
    result["GGR"] = _clean_numeric_text(result["GGR"])

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

    invalid_count = int(invalid_identifier.sum())

    if invalid_count > 0:
        if invalid_identifier_policy == "raise":
            raise ValueError(
                "Cannot construct Game_ID for "
                f"{invalid_count} rows with missing SystemName/TournamentName."
            )

        # Exclude invalid lifecycle records from modelling data.
        result = result.loc[~invalid_identifier].copy()

        system = _clean_identifier(result["SystemName"])
        tournament = _clean_identifier(result["TournamentName"])

    # Store an auditable count for the notebook/reporting layer.
    result.attrs["excluded_invalid_identifier_rows"] = invalid_count

    # Separator makes accidental concatenation collisions less likely than raw string joining.
    result["Game_ID"] = system + "::" + tournament

    # Lifecycle reference fields
    result["First_Observed_Date"] = (
        result.groupby("Game_ID")["Date"]
        .transform("min")
    )

    result["Days_Since_First_Observed"] = (
        result["Date"] - result["First_Observed_Date"]
    ).dt.days + 1

    # Recalculated GGR percentage
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