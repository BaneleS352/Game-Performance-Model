from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


REQUIRED_COLUMNS = (
    "Date",
    "SystemName",
    "TournamentName",
    "Clients",
    "BetCount",
    "Stakes",
    "GGR",
)

REQUIRED_NUMERIC_COLUMNS = ("Clients", "BetCount", "Stakes", "GGR")
OPTIONAL_CATEGORICAL_COLUMNS = (
    "Provider", "Supplier", "Studio", "GameCategory", "GameType", "Category",
    "Theme", "Volatility", "RTP", "Market", "Country", "Brand", "DeviceType",
)


@dataclass(frozen=True)
class CleaningConfig:
    """Safe, upload-independent cleaning choices for daily game records."""

    invalid_identifier_policy: str = "drop"
    missing_column_threshold: float = 0.50
    detect_outliers: bool = True
    outlier_iqr_multiplier: float = 1.5


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
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def _missing_values(series: pd.Series) -> pd.Series:
    """Treat blank strings as missing without changing genuine zero values."""
    return series.isna() | series.astype("string").str.strip().eq("")


def _outlier_rows(frame: pd.DataFrame, multiplier: float) -> tuple[pd.Series, dict[str, int]]:
    """Flag IQR outliers for review; this pipeline deliberately never drops them."""
    mask = pd.Series(False, index=frame.index)
    counts: dict[str, int] = {}
    for column in REQUIRED_NUMERIC_COLUMNS:
        values = frame[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        flagged = frame[column].lt(q1 - multiplier * iqr) | frame[column].gt(q3 + multiplier * iqr)
        counts[column] = int(flagged.sum())
        mask |= flagged
    return mask, counts


def load_raw_workbook(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Load the raw workbook from disk."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Source workbook not found: {path}. Set GAME_DATA_FILE to the correct path."
        )

    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except EmptyDataError as error:
            raise ValueError("The uploaded CSV is empty. Add a header row and daily game records.") from error
        except ParserError as error:
            raise ValueError("The uploaded CSV could not be read. Check its delimiter and quoting.") from error
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the required source columns are present."""
    source_columns: dict[str, list[object]] = {}
    for column in df.columns:
        source_columns.setdefault(str(column).strip().casefold(), []).append(column)

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
    source_columns = {str(column).strip().casefold(): column for column in df.columns}
    return df.rename(columns={source_columns[column.casefold()]: column for column in REQUIRED_COLUMNS})


def clean_game_daily(
    df: pd.DataFrame,
    invalid_identifier_policy: str = "drop",
    *,
    config: CleaningConfig | None = None,
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
    config = config or CleaningConfig(invalid_identifier_policy=invalid_identifier_policy)
    if config.invalid_identifier_policy not in {"drop", "raise"}:
        raise ValueError("invalid_identifier_policy must be 'drop' or 'raise'.")
    if not 0 <= config.missing_column_threshold <= 1 or config.outlier_iqr_multiplier <= 0:
        raise ValueError("CleaningConfig thresholds must be non-negative and valid.")

    result = _canonicalize_required_columns(df.copy())

    input_rows = len(result)
    if result.empty:
        raise ValueError("The selected worksheet is empty. Add at least one daily game record.")
    missing_values_found = int(sum(_missing_values(result[column]).sum() for column in REQUIRED_COLUMNS))

    # Dates
    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    invalid_dates = result["Date"].isna()

    # Numeric fields
    for column in REQUIRED_NUMERIC_COLUMNS:
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
        if config.invalid_identifier_policy == "raise":
            raise ValueError(
                "Cannot construct Game_ID for "
                f"{invalid_identifier_count} rows with missing SystemName/TournamentName."
            )

    invalid_numeric = result[list(REQUIRED_NUMERIC_COLUMNS)].isna().any(axis=1)
    invalid_numeric |= ~np.isfinite(result[list(REQUIRED_NUMERIC_COLUMNS)]).all(axis=1)
    negative_activity = result[["Clients", "BetCount", "Stakes"]].lt(0).any(axis=1)
    corrupted = invalid_dates | invalid_identifier | invalid_numeric | negative_activity

    # A daily game record must have one valid row per game and date. First
    # remove exact repeats, then retain the last corrected export for a key.
    valid = result.loc[~corrupted].copy()
    valid["SystemName"] = _clean_identifier(valid["SystemName"])
    valid["TournamentName"] = _clean_identifier(valid["TournamentName"])
    exact_duplicate_mask = valid.duplicated(keep="last")
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    valid = valid.loc[~exact_duplicate_mask].copy()
    valid["Game_ID"] = valid["SystemName"] + "::" + valid["TournamentName"]
    duplicate_mask = valid.duplicated(subset=["Date", "Game_ID"], keep="last")
    key_duplicate_count = int(duplicate_mask.sum())
    duplicate_count = exact_duplicate_count + key_duplicate_count
    result = valid.loc[~duplicate_mask].copy().reset_index(drop=True)

    # Optional metadata is categorical context. A stable Unknown is safer than
    # inventing a mode from all rows, which could leak holdout information.
    for column in OPTIONAL_CATEGORICAL_COLUMNS:
        if column in result:
            result[column] = _clean_identifier(result[column]).fillna("Unknown")

    outlier_mask, outlier_by_column = (
        _outlier_rows(result, config.outlier_iqr_multiplier)
        if config.detect_outliers else (pd.Series(False, index=result.index), {})
    )
    missing_columns = {
        column: float(_missing_values(df[column]).mean())
        for column in df.columns
        if len(df) and _missing_values(df[column]).mean() >= config.missing_column_threshold
    }

    # Store auditable counts for the notebook/reporting layer.
    result.attrs.update(
        {
            "input_rows": int(input_rows),
            "original_rows": int(input_rows),
            "missing_values_found": missing_values_found,
            "invalid_date_rows": int(invalid_dates.sum()),
            "excluded_invalid_identifier_rows": invalid_identifier_count,
            "invalid_numeric_rows": int(invalid_numeric.sum()),
            "negative_activity_rows": int(negative_activity.sum()),
            "corrupted_rows_removed": int(corrupted.sum()),
            "duplicate_rows_removed": duplicate_count,
            "completely_duplicated_rows_removed": exact_duplicate_count,
            "game_date_duplicates_removed": key_duplicate_count,
            "duplicates_found": duplicate_count,
            "rows_remaining_after_deduplication": int(len(result)),
            "invalid_rows": int(corrupted.sum()),
            "outliers_detected": int(outlier_mask.sum()),
            "outliers_by_column": outlier_by_column,
            "columns_with_excessive_missing_data": missing_columns,
            "columns_processed": int(len(result.columns)),
            "output_rows": int(len(result)),
            "final_rows": int(len(result)),
            "status": "success",
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
