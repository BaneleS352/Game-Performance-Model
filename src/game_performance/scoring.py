from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OBSERVATION_WINDOW_DAYS, PREDICTION_WEEK_END, REGRESSION_TARGETS, build_four_week_snapshot, prepare_features
from .modeling import confidence_from_prediction, _baseline_prediction, _prediction_from_fit


def assign_merchandising_recommendation(row: pd.Series) -> str:
    predicted_clients = row.get("predicted_clients_week_4", np.nan)
    predicted_stakes = row.get("predicted_stakes_week_4", np.nan)
    horizon = row.get("Horizon", np.nan)
    if pd.notna(predicted_clients) and pd.notna(predicted_stakes) and predicted_clients > 0 and predicted_stakes > 0:
        return "Priority placement for week 4"
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
        if target not in REGRESSION_TARGETS:
            continue
        short = target.replace("target_", "")
        transform = model_bundle.get("target_transforms", {}).get(target, "raw")
        model_pred = _prediction_from_fit(target, model.predict(x), transform)
        baseline_pred = _baseline_prediction(snapshot, target)
        blend_weight = model_bundle.get("blend_weights", {}).get(target, 1.0)
        pred = blend_weight * model_pred + (1 - blend_weight) * baseline_pred
        result[f"predicted_{short}_week_4"] = pred
        result[f"confidence_{short}"] = confidence_from_prediction(
            pred,
            model_bundle.get("confidence_tables", {}).get(target),
            snapshot.get("observation_coverage"),
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
    status = status[status["Current_Age"] >= OBSERVATION_WINDOW_DAYS].copy()
    status["Checkpoint"] = OBSERVATION_WINDOW_DAYS
    new_scores = []
    already = set()
    if not scoring_log.empty and {"Game_ID", "Checkpoint", "Horizon"}.issubset(scoring_log.columns):
        already = set(zip(scoring_log["Game_ID"], scoring_log["Checkpoint"], scoring_log["Horizon"]))
    checkpoint = OBSERVATION_WINDOW_DAYS
    horizon = PREDICTION_WEEK_END - OBSERVATION_WINDOW_DAYS
    model_bundle = model_bundles.get((checkpoint, horizon))
    if model_bundle is None:
        return pd.DataFrame(), scoring_log
    game_ids = status["Game_ID"].tolist()
    eligible_ids = [g for g in game_ids if (g, checkpoint, horizon) not in already]
    if eligible_ids:
        snapshot = build_four_week_snapshot(history[history["Game_ID"].isin(eligible_ids)], require_future=False)
        scores = score_snapshot(snapshot, model_bundle)
        scores["Checkpoint"] = checkpoint
        scores["Horizon"] = horizon
        scores["As_Of_Date"] = as_of_date
        new_scores.append(scores)
    if not new_scores: return pd.DataFrame(), scoring_log
    predictions = pd.concat(new_scores, ignore_index=True)
    log_addition = predictions[["Game_ID", "Checkpoint", "Horizon", "As_Of_Date", "scored_at"]]
    updated = pd.concat([scoring_log, log_addition], ignore_index=True).drop_duplicates(
        ["Game_ID", "Checkpoint", "Horizon"], keep="last"
    )
    return predictions, updated
