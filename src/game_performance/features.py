from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

OPTIONAL_METADATA_CANDIDATES = [
    "Provider", "Supplier", "Studio", "GameCategory", "GameType", "Category",
    "Theme", "Volatility", "RTP", "Market", "Country", "Brand", "DeviceType",
]

REGRESSION_TARGETS = ["target_clients", "target_stakes"]
OBSERVATION_WINDOW_DAYS = 14
VALIDATION_WEEK_START = 15
VALIDATION_WEEK_END = 21
PREDICTION_WEEK_START = 22
PREDICTION_WEEK_END = 28

BASE_CATEGORICAL_FEATURES = [
    "SystemName",
    "TournamentName",
    "Launch_Month",
    "Launch_Weekday",
    "Observation_End_Month",
    "Observation_End_Weekday",
    "Clients_Size_Band",
    "Stakes_Size_Band",
]
BASE_NUMERIC_FEATURES = [
    "Obs_Window", "Horizon",
    "Observation_End_DayOfMonth", "Observation_End_WeekOfYear",
    "obs_clients_sum", "obs_stakes_sum",
    "clients_week_1_sum", "clients_week_2_sum", "clients_week_1_to_2_growth",
    "stakes_week_1_sum", "stakes_week_2_sum", "stakes_week_1_to_2_growth",
    "stake_per_client_week_1", "stake_per_client_week_2",
    "log_obs_clients_sum", "log_obs_stakes_sum", "log_clients_week_2_sum", "log_stakes_week_2_sum",
    "obs_clients_mean", "obs_stakes_mean",
    "obs_clients_std", "obs_stakes_std",
    "clients_first", "stakes_first",
    "clients_last", "stakes_last",
    "clients_median", "stakes_median",
    "clients_cv", "stakes_cv",
    "stake_per_client",
    "clients_slope", "stakes_slope",
    "clients_growth", "stakes_growth",
    "clients_last_3_mean", "stakes_last_3_mean",
    "clients_last_7_mean", "stakes_last_7_mean",
    "clients_recent_to_early_ratio", "stakes_recent_to_early_ratio",
    "clients_recent_share", "stakes_recent_share",
    "clients_peak", "stakes_peak",
    "days_since_clients_peak", "days_since_stakes_peak",
    "clients_zero_days", "stakes_zero_days",
    "clients_positive_days", "stakes_positive_days",
    "active_observation_days", "observation_coverage",
]


def available_optional_metadata(df: pd.DataFrame) -> list[str]:
    return [c for c in OPTIONAL_METADATA_CANDIDATES if c in df.columns]


def categorical_features(df: pd.DataFrame) -> list[str]:
    return BASE_CATEGORICAL_FEATURES + available_optional_metadata(df)


def missing_indicator_features() -> list[str]:
    return [f"missing__{column}" for column in BASE_NUMERIC_FEATURES]


def model_features(df: pd.DataFrame) -> list[str]:
    return categorical_features(df) + BASE_NUMERIC_FEATURES + missing_indicator_features()


def _safe_divide(numerator: float, denominator: float) -> float:
    if numerator is None or pd.isna(numerator) or denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _linear_slope(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series), dtype=float)
    return float(np.polyfit(x, series.to_numpy(), 1)[0])


def _first_last_growth(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return np.nan
    split = max(1, len(series) // 2)
    first_mean = series.iloc[:split].mean()
    last_mean = series.iloc[split:].mean()
    if pd.isna(first_mean) or first_mean == 0:
        return np.nan
    return float((last_mean - first_mean) / abs(first_mean))


def _size_band(value: float, thresholds: tuple[float, float, float]) -> str:
    if pd.isna(value) or value <= 0:
        return "No activity"
    if value < thresholds[0]:
        return "Small"
    if value < thresholds[1]:
        return "Medium"
    if value < thresholds[2]:
        return "Large"
    return "Very large"


def _safe_log1p(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.log1p(max(value, 0)))


def _recent_mean(values: Iterable[float], window: int) -> float:
    series = pd.Series(values, dtype="float64")
    return np.nan if series.notna().sum() == 0 else float(series.tail(window).mean())


def _recent_to_early_ratio(values: Iterable[float], window: int = 3) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return np.nan
    early = series.head(min(window, len(series))).mean()
    recent = series.tail(min(window, len(series))).mean()
    return _safe_divide(recent, abs(early))


def _coefficient_of_variation(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return np.nan
    mean = series.mean()
    if pd.isna(mean) or mean == 0:
        return np.nan
    return float(series.std(ddof=0) / abs(mean))


def _days_since_peak(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype="float64").reset_index(drop=True)
    if series.notna().sum() == 0:
        return np.nan
    peak_position = int(series.idxmax())
    return float(len(series) - 1 - peak_position)


def build_lifecycle_snapshot(
    data: pd.DataFrame,
    obs_window: int,
    horizon: int,
    *,
    require_future: bool = True,
) -> pd.DataFrame:
    """Build one point-in-time row per game for a checkpoint/horizon pair."""
    required = {
        "Game_ID", "Date", "Days_Since_First_Observed", "First_Observed_Date",
        "SystemName", "TournamentName", "Clients", "Stakes",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    optional_metadata = available_optional_metadata(data)
    rows: list[dict[str, object]] = []

    for game_id, game in data.groupby("Game_ID", sort=False):
        game = game.sort_values("Date").copy()
        max_age = game["Days_Since_First_Observed"].max()
        if pd.isna(max_age) or max_age < obs_window:
            continue
        if require_future and max_age < obs_window + horizon:
            continue

        obs = game[game["Days_Since_First_Observed"].between(1, obs_window)].copy()
        future = game[game["Days_Since_First_Observed"].between(obs_window + 1, obs_window + horizon)].copy()
        if obs.empty or (require_future and future.empty):
            continue

        daily_obs = (
            obs.groupby("Date", as_index=False)
            .agg({"Clients": "sum", "Stakes": "sum"})
            .sort_values("Date")
            .reset_index(drop=True)
        )
        first_observed = pd.to_datetime(game["First_Observed_Date"].iloc[0])
        observation_end = first_observed + pd.to_timedelta(obs_window - 1, unit="D")
        row: dict[str, object] = {
            "Game_ID": game_id,
            "SystemName": obs["SystemName"].iloc[0],
            "TournamentName": obs["TournamentName"].iloc[0],
            "First_Observed_Date": first_observed,
            "Launch_Month": first_observed.strftime("%B") if pd.notna(first_observed) else "Unknown",
            "Launch_Weekday": first_observed.strftime("%A") if pd.notna(first_observed) else "Unknown",
            "Observation_End_Month": observation_end.strftime("%B") if pd.notna(observation_end) else "Unknown",
            "Observation_End_Weekday": observation_end.strftime("%A") if pd.notna(observation_end) else "Unknown",
            "Observation_End_DayOfMonth": observation_end.day if pd.notna(observation_end) else np.nan,
            "Observation_End_WeekOfYear": (
                int(observation_end.isocalendar().week) if pd.notna(observation_end) else np.nan
            ),
            "Obs_Window": obs_window,
            "Horizon": horizon,
            "Lifecycle_Stage": f"Day {obs_window}",
            "active_observation_days": int(daily_obs["Date"].nunique()),
            "observation_coverage": daily_obs["Date"].nunique() / obs_window,
        }

        for column in optional_metadata:
            non_missing = obs[column].dropna()
            row[column] = non_missing.iloc[0] if not non_missing.empty else "Unknown"

        for source, prefix in (("Clients", "clients"), ("Stakes", "stakes")):
            series = daily_obs[source]
            total = series.sum(min_count=1)
            row[f"obs_{prefix}_sum"] = total
            row[f"obs_{prefix}_mean"] = series.mean()
            row[f"obs_{prefix}_std"] = series.std(ddof=0)
            row[f"{prefix}_first"] = series.iloc[0] if not series.empty else np.nan
            row[f"{prefix}_last"] = series.iloc[-1] if not series.empty else np.nan
            row[f"{prefix}_median"] = series.median()
            row[f"{prefix}_cv"] = _coefficient_of_variation(series)
            row[f"{prefix}_slope"] = _linear_slope(series)
            row[f"{prefix}_growth"] = _first_last_growth(series)
            row[f"{prefix}_last_3_mean"] = _recent_mean(series, 3)
            row[f"{prefix}_last_7_mean"] = _recent_mean(series, 7)
            row[f"{prefix}_recent_to_early_ratio"] = _recent_to_early_ratio(series, 3)
            row[f"{prefix}_recent_share"] = _safe_divide(series.tail(3).sum(min_count=1), total)
            row[f"{prefix}_peak"] = series.max()
            row[f"days_since_{prefix}_peak"] = _days_since_peak(series)
            row[f"{prefix}_zero_days"] = int(series.fillna(0).eq(0).sum())
            row[f"{prefix}_positive_days"] = int(series.gt(0).sum())

        row["stake_per_client"] = _safe_divide(row["obs_stakes_sum"], row["obs_clients_sum"])

        week_1_obs = obs[obs["Days_Since_First_Observed"].between(1, 7)]
        week_2_obs = obs[obs["Days_Since_First_Observed"].between(8, obs_window)]
        clients_week_1 = week_1_obs["Clients"].sum(min_count=1)
        clients_week_2 = week_2_obs["Clients"].sum(min_count=1)
        stakes_week_1 = week_1_obs["Stakes"].sum(min_count=1)
        stakes_week_2 = week_2_obs["Stakes"].sum(min_count=1)

        row["clients_week_1_sum"] = clients_week_1
        row["clients_week_2_sum"] = clients_week_2
        row["clients_week_1_to_2_growth"] = _safe_divide(clients_week_2 - clients_week_1, abs(clients_week_1))
        row["stakes_week_1_sum"] = stakes_week_1
        row["stakes_week_2_sum"] = stakes_week_2
        row["stakes_week_1_to_2_growth"] = _safe_divide(stakes_week_2 - stakes_week_1, abs(stakes_week_1))
        row["stake_per_client_week_1"] = _safe_divide(stakes_week_1, clients_week_1)
        row["stake_per_client_week_2"] = _safe_divide(stakes_week_2, clients_week_2)
        row["log_obs_clients_sum"] = _safe_log1p(row["obs_clients_sum"])
        row["log_obs_stakes_sum"] = _safe_log1p(row["obs_stakes_sum"])
        row["log_clients_week_2_sum"] = _safe_log1p(clients_week_2)
        row["log_stakes_week_2_sum"] = _safe_log1p(stakes_week_2)
        row["Clients_Size_Band"] = _size_band(row["obs_clients_sum"], (100, 1_000, 10_000))
        row["Stakes_Size_Band"] = _size_band(row["obs_stakes_sum"], (10_000, 100_000, 1_000_000))

        if require_future:
            # Future rows are targets only; no future-derived column is used as a feature.
            week_1 = game[game["Days_Since_First_Observed"].between(obs_window + 1, obs_window + 7)].copy()
            week_2 = game[game["Days_Since_First_Observed"].between(obs_window + 8, obs_window + 14)].copy()
            row["target_clients"] = future["Clients"].sum(min_count=1)
            row["target_stakes"] = future["Stakes"].sum(min_count=1)
            row["target_clients_week_1"] = week_1["Clients"].sum(min_count=1)
            row["target_clients_week_2"] = week_2["Clients"].sum(min_count=1)
            row["target_stakes_week_1"] = week_1["Stakes"].sum(min_count=1)
            row["target_stakes_week_2"] = week_2["Stakes"].sum(min_count=1)

        rows.append(row)

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def build_four_week_snapshot(
    data: pd.DataFrame,
    *,
    require_future: bool = True,
) -> pd.DataFrame:
    """Build one fixed 4-week row per game.

    Weeks 1-2 (days 1-14) are features. Week 3 is validation context.
    Week 4 is the prediction target when future data is required.
    """
    snapshot = build_lifecycle_snapshot(
        data,
        OBSERVATION_WINDOW_DAYS,
        PREDICTION_WEEK_END - OBSERVATION_WINDOW_DAYS,
        require_future=require_future,
    )
    if snapshot.empty:
        return snapshot

    snapshot["Lifecycle_Stage"] = "Week 2"
    snapshot["Prediction_Week"] = 4

    if require_future:
        actuals = data[data["Game_ID"].isin(snapshot["Game_ID"])].copy()
        actuals["week_number"] = ((actuals["Days_Since_First_Observed"] - 1) // 7 + 1).astype("int64")
        weekly = (
            actuals[actuals["week_number"].isin([3, 4])]
            .groupby(["Game_ID", "week_number"], as_index=False)
            .agg({"Clients": "sum", "Stakes": "sum"})
        )
        for metric, prefix in (("Clients", "clients"), ("Stakes", "stakes")):
            wide = weekly.pivot(index="Game_ID", columns="week_number", values=metric)
            snapshot[f"actual_{prefix}_week_3"] = snapshot["Game_ID"].map(wide.get(3))
            snapshot[f"actual_{prefix}_week_4"] = snapshot["Game_ID"].map(wide.get(4))
            snapshot[f"target_{prefix}"] = snapshot[f"actual_{prefix}_week_4"]

    return snapshot.replace([np.inf, -np.inf], np.nan)


def prepare_features(frame: pd.DataFrame, categorical: list[str] | None = None) -> pd.DataFrame:
    categorical = categorical or categorical_features(frame)
    result = frame.copy()
    missing = {}
    for column in BASE_NUMERIC_FEATURES:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
        missing[f"missing__{column}"] = result[column].isna().astype("int8")
    for column in categorical:
        if column not in result.columns:
            result[column] = "Unknown"
        result[column] = result[column].astype("string").fillna("Unknown")
    result = pd.concat([result, pd.DataFrame(missing, index=result.index)], axis=1)
    return result[categorical + BASE_NUMERIC_FEATURES + missing_indicator_features()]


def build_all_historical_snapshots(data: pd.DataFrame, strategy: dict[int, list[int]]) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (OBSERVATION_WINDOW_DAYS, PREDICTION_WEEK_END - OBSERVATION_WINDOW_DAYS): build_four_week_snapshot(
            data,
            require_future=True,
        )
    }
