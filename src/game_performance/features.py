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

BASE_CATEGORICAL_FEATURES = ["SystemName", "TournamentName", "Launch_Month", "Launch_Weekday"]
BASE_NUMERIC_FEATURES = [
    "Obs_Window", "Horizon",
    "obs_clients_sum", "obs_betcount_sum", "obs_stakes_sum", "obs_ggr_sum", "obs_ggr_pct",
    "obs_clients_mean", "obs_betcount_mean", "obs_stakes_mean", "obs_ggr_mean",
    "obs_clients_std", "obs_betcount_std", "obs_stakes_std", "obs_ggr_std",
    "bets_per_client", "stake_per_client", "ggr_per_client",
    "clients_slope", "betcount_slope", "stakes_slope", "ggr_slope",
    "clients_growth", "betcount_growth", "stakes_growth", "ggr_growth",
    "clients_last_3_mean", "betcount_last_3_mean", "stakes_last_3_mean", "ggr_last_3_mean",
    "clients_last_7_mean", "betcount_last_7_mean", "stakes_last_7_mean", "ggr_last_7_mean",
    "clients_recent_share", "betcount_recent_share", "stakes_recent_share", "ggr_recent_share",
    "clients_peak", "betcount_peak", "stakes_peak", "ggr_peak",
    "days_since_clients_peak", "days_since_betcount_peak", "days_since_stakes_peak", "days_since_ggr_peak",
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
    if denominator is None or pd.isna(denominator) or denominator == 0:
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


def _recent_mean(values: Iterable[float], window: int) -> float:
    series = pd.Series(values, dtype="float64")
    return np.nan if series.notna().sum() == 0 else float(series.tail(window).mean())


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
        "SystemName", "TournamentName", "Clients", "Betcount", "Stakes", "GGR",
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
            .agg({"Clients": "sum", "Betcount": "sum", "Stakes": "sum", "GGR": "sum"})
            .sort_values("Date")
            .reset_index(drop=True)
        )
        first_observed = pd.to_datetime(game["First_Observed_Date"].iloc[0])
        row: dict[str, object] = {
            "Game_ID": game_id,
            "SystemName": obs["SystemName"].iloc[0],
            "TournamentName": obs["TournamentName"].iloc[0],
            "First_Observed_Date": first_observed,
            "Launch_Month": first_observed.strftime("%B") if pd.notna(first_observed) else "Unknown",
            "Launch_Weekday": first_observed.strftime("%A") if pd.notna(first_observed) else "Unknown",
            "Obs_Window": obs_window,
            "Horizon": horizon,
            "Lifecycle_Stage": f"Day {obs_window}",
            "active_observation_days": int(daily_obs["Date"].nunique()),
            "observation_coverage": daily_obs["Date"].nunique() / obs_window,
        }

        for column in optional_metadata:
            non_missing = obs[column].dropna()
            row[column] = non_missing.iloc[0] if not non_missing.empty else "Unknown"

        for source, prefix in (("Clients", "clients"), ("Betcount", "betcount"), ("Stakes", "stakes"), ("GGR", "ggr")):
            series = daily_obs[source]
            total = series.sum(min_count=1)
            row[f"obs_{prefix}_sum"] = total
            row[f"obs_{prefix}_mean"] = series.mean()
            row[f"obs_{prefix}_std"] = series.std(ddof=0)
            row[f"{prefix}_slope"] = _linear_slope(series)
            row[f"{prefix}_growth"] = _first_last_growth(series)
            row[f"{prefix}_last_3_mean"] = _recent_mean(series, 3)
            row[f"{prefix}_last_7_mean"] = _recent_mean(series, 7)
            row[f"{prefix}_recent_share"] = _safe_divide(series.tail(3).sum(min_count=1), total)
            row[f"{prefix}_peak"] = series.max()
            row[f"days_since_{prefix}_peak"] = _days_since_peak(series)

        row["obs_ggr_pct"] = _safe_divide(row["obs_ggr_sum"], row["obs_stakes_sum"])
        row["bets_per_client"] = _safe_divide(row["obs_betcount_sum"], row["obs_clients_sum"])
        row["stake_per_client"] = _safe_divide(row["obs_stakes_sum"], row["obs_clients_sum"])
        row["ggr_per_client"] = _safe_divide(row["obs_ggr_sum"], row["obs_clients_sum"])

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
    for column in BASE_NUMERIC_FEATURES:
        if column not in result.columns:
            result[column] = np.nan
        result[f"missing__{column}"] = result[column].isna().astype("int8")
    for column in categorical:
        if column not in result.columns:
            result[column] = "Unknown"
        result[column] = result[column].astype("string").fillna("Unknown")
    return result[categorical + BASE_NUMERIC_FEATURES + missing_indicator_features()]


def build_all_historical_snapshots(data: pd.DataFrame, strategy: dict[int, list[int]]) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (OBSERVATION_WINDOW_DAYS, PREDICTION_WEEK_END - OBSERVATION_WINDOW_DAYS): build_four_week_snapshot(
            data,
            require_future=True,
        )
    }
