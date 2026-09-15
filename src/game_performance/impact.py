from __future__ import annotations

import numpy as np
import pandas as pd


IMPACT_PERIOD_COLUMNS = [
    "Game_ID",
    "SystemName",
    "TournamentName",
    "Start_Date",
    "End_Date",
    "Metric",
    "Direction",
    "Baseline_Value",
    "Observed_Value",
    "Percent_Change",
    "Severity",
    "Reason",
]
IMPACT_METRICS = ("Clients", "BetCount", "Stakes", "GGR")


def detect_external_impact_periods(
    cleaned: pd.DataFrame,
    *,
    window: int = 7,
    minimum_history: int = 3,
    change_threshold: float = 0.50,
) -> pd.DataFrame:
    """Infer likely promo/external-impact periods from cleaned daily data."""
    if cleaned.empty:
        return pd.DataFrame(columns=IMPACT_PERIOD_COLUMNS)

    required = {"Game_ID", "Date", *IMPACT_METRICS}
    missing = required.difference(cleaned.columns)
    if missing:
        raise KeyError(f"Missing required impact columns: {sorted(missing)}")

    frame = cleaned.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Game_ID", "Date"])
    rows: list[dict] = []

    for game_id, group in frame.sort_values(["Game_ID", "Date"]).groupby("Game_ID", sort=False):
        group = group.drop_duplicates("Date", keep="last").reset_index(drop=True)
        for metric in IMPACT_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            baseline = values.shift(1).rolling(window, min_periods=minimum_history).median()
            valid = baseline.gt(0) & values.notna()
            change = (values - baseline) / baseline.where(valid)
            flags = valid & change.abs().ge(change_threshold)
            for index in np.flatnonzero(flags.to_numpy()):
                direction = "Spike" if change.iloc[index] > 0 else "Drop"
                rows.append(
                    {
                        "Game_ID": game_id,
                        "SystemName": group.loc[index, "SystemName"],
                        "TournamentName": group.loc[index, "TournamentName"],
                        "Date": group.loc[index, "Date"],
                        "Metric": metric,
                        "Direction": direction,
                        "Baseline_Value": float(baseline.iloc[index]),
                        "Observed_Value": float(values.iloc[index]),
                        "Percent_Change": float(change.iloc[index] * 100),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=IMPACT_PERIOD_COLUMNS)

    flagged = pd.DataFrame(rows).sort_values(["Game_ID", "Metric", "Direction", "Date"])
    periods: list[dict] = []
    for keys, group in flagged.groupby(["Game_ID", "Metric", "Direction"], sort=False):
        group = group.reset_index(drop=True)
        breaks = group["Date"].diff().dt.days.ne(1).cumsum()
        for _, period in group.groupby(breaks, sort=False):
            baseline_value = float(period["Baseline_Value"].mean())
            observed_value = float(period["Observed_Value"].mean())
            percent_change = (observed_value - baseline_value) / baseline_value * 100
            severity = "High" if abs(percent_change) >= 100 else "Medium"
            direction = keys[2]
            periods.append(
                {
                    "Game_ID": keys[0],
                    "SystemName": period["SystemName"].iloc[0],
                    "TournamentName": period["TournamentName"].iloc[0],
                    "Start_Date": period["Date"].min(),
                    "End_Date": period["Date"].max(),
                    "Metric": keys[1],
                    "Direction": direction,
                    "Baseline_Value": baseline_value,
                    "Observed_Value": observed_value,
                    "Percent_Change": percent_change,
                    "Severity": severity,
                    "Reason": f"Inferred external impact: {direction.lower()} in {keys[1]} versus recent baseline.",
                }
            )

    return pd.DataFrame(periods, columns=IMPACT_PERIOD_COLUMNS)


def add_external_impact_flags(
    forecasts: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """Add compact, explanatory impact fields to forecasts with recent periods."""
    result = forecasts.copy()
    if result.empty or periods.empty or "Game_ID" not in result.columns:
        return result

    impact = periods.copy()
    impact["Start_Date"] = pd.to_datetime(impact["Start_Date"], errors="coerce")
    impact["End_Date"] = pd.to_datetime(impact["End_Date"], errors="coerce")
    as_of = pd.to_datetime(result.get("As_Of_Date"), errors="coerce")
    if as_of is None or not isinstance(as_of, pd.Series):
        as_of = pd.Series(pd.Timestamp.max, index=result.index)

    def matches(row: pd.Series) -> pd.DataFrame:
        cutoff = row["as_of"] - pd.Timedelta(days=lookback_days)
        return impact[(impact["Game_ID"] == row["Game_ID"]) & (impact["End_Date"] >= cutoff) & (impact["Start_Date"] <= row["as_of"])]

    result["as_of"] = as_of.reindex(result.index).fillna(pd.Timestamp.max)
    matched = result.apply(matches, axis=1)
    result["external_impact_detected"] = matched.map(lambda frame: bool(len(frame)))
    result["external_impact_periods"] = matched.map(
        lambda frame: "; ".join(
            f"{row.Start_Date:%Y-%m-%d} to {row.End_Date:%Y-%m-%d} ({row.Metric} {row.Direction.lower()})"
            for row in frame.itertuples()
        )
    )
    result["external_impact_reason"] = matched.map(
        lambda frame: "; ".join(frame["Reason"].astype(str))
    )
    return result.drop(columns=["as_of"])
