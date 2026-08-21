from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_lifecycle_snapshot, prepare_features


def assign_merchandising_recommendation(row: pd.Series) -> str:
    probability = row.get("probability_high_performance", np.nan)
    predicted_ggr = row.get("predicted_ggr", np.nan)
    predicted_margin = row.get("derived_predicted_ggr_pct", np.nan)
    horizon = row.get("Horizon", np.nan)
    if pd.notna(probability) and probability >= 0.80 and predicted_ggr > 0 and predicted_margin >= 0:
        return f"Priority placement for next {int(horizon)} days"
    if pd.notna(probability) and probability >= 0.60 and predicted_ggr > 0:
        return f"Increase visibility for next {int(horizon)} days"
    if pd.notna(probability) and probability >= 0.40:
        return "Standard placement with monitoring"
    if pd.notna(predicted_ggr) and predicted_ggr < 0:
        return "Do not promote; review negative GGR risk"
    return "Standard catalogue placement"


def score_snapshot(snapshot: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot.copy()
    categorical = model_bundle["categorical_features"]
    x = prepare_features(snapshot, categorical)
    cols = ["Game_ID", "TournamentName", "SystemName", "First_Observed_Date",
            "Lifecycle_Stage", "Obs_Window", "Horizon", "active_observation_days", "observation_coverage"]
    result = snapshot[cols].copy()
    for target, model in model_bundle["regressors"].items():
        short = target.replace("target_", "")
        pred = model.predict(x)
        if target in {"target_clients", "target_betcount", "target_stakes"}:
            pred = np.maximum(pred, 0)
        result[f"predicted_{short}"] = pred
    classifier = model_bundle.get("classifier")
    if classifier is not None:
        result["probability_high_performance"] = classifier.predict_proba(x)[:, 1]
        result["predicted_high_performer"] = (result["probability_high_performance"] >= 0.50).astype(int)
    if {"predicted_ggr", "predicted_stakes"}.issubset(result.columns):
        result["derived_predicted_ggr_pct"] = np.where(
            result["predicted_stakes"] > 0, result["predicted_ggr"] / result["predicted_stakes"], np.nan
        )
    result["recommendation"] = result.apply(assign_merchandising_recommendation, axis=1)
    result["scored_at"] = pd.Timestamp.now(tz="UTC")
    return result


def eligible_checkpoint_for_age(age: float, checkpoints: list[int], exact_checkpoint_only: bool = True):
    if pd.isna(age): return None
    age = int(age)
    if exact_checkpoint_only: return age if age in checkpoints else None
    eligible = [c for c in checkpoints if c <= age]
    return max(eligible) if eligible else None


def run_daily_lifecycle_scoring(data, model_bundles, horizon_strategy, *, as_of_date=None,
                                scoring_log=None, exact_checkpoint_only=True):
    as_of_date = pd.Timestamp(data["Date"].max() if as_of_date is None else as_of_date).normalize()
    history = data[data["Date"] <= as_of_date].copy()
    scoring_log = pd.DataFrame() if scoring_log is None else scoring_log.copy()
    status = history.groupby("Game_ID", as_index=False).agg(Current_Age=("Days_Since_First_Observed", "max"))
    checkpoints = sorted(horizon_strategy)
    status["Checkpoint"] = status["Current_Age"].apply(
        lambda age: eligible_checkpoint_for_age(age, checkpoints, exact_checkpoint_only)
    )
    status = status.dropna(subset=["Checkpoint"]); status["Checkpoint"] = status["Checkpoint"].astype(int)
    new_scores = []
    already = set()
    if not scoring_log.empty and {"Game_ID", "Checkpoint", "Horizon"}.issubset(scoring_log.columns):
        already = set(zip(scoring_log["Game_ID"], scoring_log["Checkpoint"], scoring_log["Horizon"]))
    for checkpoint in sorted(status["Checkpoint"].unique()):
        game_ids = status.loc[status["Checkpoint"] == checkpoint, "Game_ID"].tolist()
        for horizon in sorted(h for c, h in model_bundles if c == checkpoint):
            eligible_ids = [g for g in game_ids if (g, checkpoint, horizon) not in already]
            if not eligible_ids: continue
            snapshot = build_lifecycle_snapshot(
                history[history["Game_ID"].isin(eligible_ids)], checkpoint, horizon, require_future=False
            )
            scores = score_snapshot(snapshot, model_bundles[(checkpoint, horizon)])
            scores["Checkpoint"] = checkpoint; scores["As_Of_Date"] = as_of_date
            new_scores.append(scores)
    if not new_scores: return pd.DataFrame(), scoring_log
    predictions = pd.concat(new_scores, ignore_index=True)
    log_addition = predictions[["Game_ID", "Checkpoint", "Horizon", "As_Of_Date", "scored_at"]]
    updated = pd.concat([scoring_log, log_addition], ignore_index=True).drop_duplicates(
        ["Game_ID", "Checkpoint", "Horizon"], keep="last"
    )
    return predictions, updated
