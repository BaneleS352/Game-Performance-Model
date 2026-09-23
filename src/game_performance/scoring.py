from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    COMMERCIAL_DRIVER_TARGETS,
    OUTCOME_TARGETS,
    build_lifecycle_snapshot,
    prepare_features,
)

from .modeling import (
    NON_NEGATIVE_TARGETS,
    _baseline_prediction,
    _prediction_from_fit,
    apply_probability_calibration,
    confidence_from_prediction,
    derive_commercial_outputs,
    prediction_interval_from_table,
)

MIN_HIGH_PERFORMER_THRESHOLD = 0.80


def _confidence_for_output(
    prediction: np.ndarray,
    target: str,
    snapshot: pd.DataFrame,
    model_bundle: dict,
) -> np.ndarray:
    confidence_table = model_bundle.get("confidence_tables", {}).get(target)
    fallback = model_bundle.get("confidence_scores", {}).get(target)
    if not confidence_table:
        confidence_table = {"fallback": fallback, "bins": []}
    elif pd.isna(confidence_table.get("fallback", np.nan)) and fallback is not None:
        confidence_table = {**confidence_table, "fallback": fallback}
    return confidence_from_prediction(prediction, confidence_table, snapshot.get("observation_coverage"))


def _add_review_flags(result: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    confidence_limit = float(model_bundle.get("review_confidence_threshold", 0.50))
    width_limit = float(model_bundle.get("review_relative_width_threshold", 1.00))
    reasons: list[str] = []
    priorities: list[float] = []
    for row in result.itertuples(index=False):
        row_data = row._asdict()
        flags: list[str] = []
        severity = 0.0
        for metric in ("clients", "stakes"):
            confidence = pd.to_numeric(pd.Series([row_data.get(f"confidence_{metric}")]), errors="coerce").iloc[0]
            prediction = pd.to_numeric(pd.Series([row_data.get(f"predicted_{metric}")]), errors="coerce").iloc[0]
            low = pd.to_numeric(pd.Series([row_data.get(f"predicted_{metric}_low")]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([row_data.get(f"predicted_{metric}_high")]), errors="coerce").iloc[0]
            if pd.notna(confidence) and confidence < confidence_limit:
                flags.append(f"low {metric} confidence")
                severity = max(severity, (confidence_limit - confidence) / max(confidence_limit, 1e-9))
            if pd.notna(prediction) and pd.notna(low) and pd.notna(high):
                relative_width = (high - low) / max(abs(prediction), 1.0)
                if relative_width > width_limit:
                    flags.append(f"wide {metric} interval")
                    severity = max(severity, relative_width / width_limit)
        reasons.append("; ".join(flags))
        priorities.append(float(severity))
    result["requires_review"] = pd.Series(reasons, index=result.index).ne("")
    result["review_reason"] = reasons
    result["review_priority"] = priorities
    return result


def assign_merchandising_recommendation(
    row: pd.Series,
) -> str:

    probability = row.get(
        "probability_high_performance",
        np.nan,
    )

    predicted_clients = row.get(
        "predicted_clients",
        np.nan,
    )

    predicted_stakes = row.get(
        "predicted_stakes",
        np.nan,
    )

    has_positive_activity = (
        pd.notna(predicted_clients)
        and pd.notna(predicted_stakes)
        and predicted_clients > 0
        and predicted_stakes > 0
    )

    if (
        pd.notna(probability)
        and probability >= 0.80
        and has_positive_activity
    ):
        return "Priority placement"

    if (
        pd.notna(probability)
        and probability >= 0.60
        and pd.notna(predicted_stakes)
        and predicted_stakes > 0
    ):
        return "Increase visibility"

    if (
        pd.notna(probability)
        and probability >= 0.40
    ):
        return "Standard placement with monitoring"

    if (
        pd.notna(predicted_stakes)
        and predicted_stakes <= 0
    ):
        return "Do not promote; review low stakes forecast"

    return "Standard catalogue placement"


def score_snapshot(
    snapshot: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:

    if snapshot.empty:
        return snapshot.copy()

    categorical = model_bundle[
        "categorical_features"
    ]

    x = prepare_features(
        snapshot,
        categorical,
    )

    cols = [
        "Game_ID",
        "TournamentName",
        "SystemName",
        "First_Observed_Date",
        "Lifecycle_Stage",
        "Prediction_Start",
        "Prediction_End",
        "Obs_Window",
        "Horizon",
        "active_observation_days",
        "observation_coverage",
    ]

    result = snapshot[[column for column in cols if column in snapshot.columns]].copy()

    # =========================================================
    # REGRESSION FORECASTS
    # =========================================================

    regressors = model_bundle.get("regressors", {})
    commercial_model = (
        model_bundle.get("forecast_strategy") == "commercial_drivers_v1"
        or set(COMMERCIAL_DRIVER_TARGETS).issubset(regressors)
    )

    if commercial_model:
        driver_predictions: dict[str, np.ndarray] = {}
        driver_baselines: dict[str, np.ndarray] = {}
        for target in COMMERCIAL_DRIVER_TARGETS:
            model = regressors[target]
            transform = model_bundle.get("target_transforms", {}).get(
                target,
                model_bundle.get("regression_target_transforms", {}).get(target, "raw"),
            )
            model_prediction = _prediction_from_fit(target, model.predict(x), transform)
            baseline = _baseline_prediction(snapshot, target)
            weight = model_bundle.get("blend_weights", {}).get(target, 1.0)
            driver_predictions[target] = weight * model_prediction + (1 - weight) * baseline
            driver_baselines[target] = baseline

        outcomes = derive_commercial_outputs(driver_predictions)
        baselines = derive_commercial_outputs(driver_baselines)
        for target in OUTCOME_TARGETS:
            short = target.removeprefix("target_")
            prediction = outcomes[target]
            result[f"predicted_{short}"] = prediction
            result[f"baseline_{short}"] = baselines[target]
            result[f"predicted_{short}_low"], result[f"predicted_{short}_high"] = prediction_interval_from_table(
                prediction,
                model_bundle.get("prediction_interval_tables", {}).get(target),
                non_negative=target in NON_NEGATIVE_TARGETS,
            )
            result[f"confidence_{short}"] = _confidence_for_output(prediction, target, snapshot, model_bundle)
    else:
        # Keep existing direct-outcome bundles scoreable after the driver-model migration.
        for target, model in regressors.items():
            if target not in OUTCOME_TARGETS:
                continue
            short = target.removeprefix("target_")
            transform = model_bundle.get("target_transforms", {}).get(
                target,
                model_bundle.get("regression_target_transforms", {}).get(target, "raw"),
            )
            model_prediction = _prediction_from_fit(target, model.predict(x), transform)
            baseline = _baseline_prediction(snapshot, target)
            weight = model_bundle.get("blend_weights", {}).get(target, 1.0)
            prediction = weight * model_prediction + (1 - weight) * baseline
            result[f"predicted_{short}"] = prediction
            result[f"baseline_{short}"] = baseline
            result[f"predicted_{short}_low"], result[f"predicted_{short}_high"] = prediction_interval_from_table(
                prediction,
                model_bundle.get("prediction_interval_tables", {}).get(target),
                non_negative=target in NON_NEGATIVE_TARGETS,
            )
            result[f"confidence_{short}"] = _confidence_for_output(prediction, target, snapshot, model_bundle)

    # =========================================================
    # HIGH-PERFORMER CLASSIFIER
    # =========================================================

    classifiers = model_bundle.get("classifiers", {})
    if not classifiers and model_bundle.get("classifier") is not None:
        classifiers = {"performance": model_bundle["classifier"]}

    for goal, classifier in classifiers.items():
        raw_probability = classifier.predict_proba(x)[:, 1]
        calibrator = (
            model_bundle
            .get("probability_calibrators", {})
            .get(goal, model_bundle.get("probability_calibrator"))
        )
        probability = apply_probability_calibration(raw_probability, calibrator)
        threshold = float(
            model_bundle
            .get("classification_thresholds", {})
            .get(goal, model_bundle.get("classification_threshold", MIN_HIGH_PERFORMER_THRESHOLD))
        )
        threshold = max(MIN_HIGH_PERFORMER_THRESHOLD, threshold)

        if goal == "performance":
            raw_column = "raw_probability_high_performance"
            probability_column = "probability_high_performance"
            threshold_column = "classification_threshold"
            predicted_column = "predicted_high_performer"
        else:
            raw_column = f"raw_probability_high_{goal}_performer"
            probability_column = f"probability_high_{goal}_performer"
            threshold_column = f"classification_threshold_high_{goal}_performer"
            predicted_column = f"predicted_high_{goal}_performer"

        result[raw_column] = raw_probability
        result[probability_column] = probability
        result[threshold_column] = threshold
        result[predicted_column] = (probability >= threshold).astype(int)

    if {"probability_high_client_performer", "probability_high_stakes_performer"}.issubset(result.columns):
        result["probability_high_performance"] = result[
            ["probability_high_client_performer", "probability_high_stakes_performer"]
        ].min(axis=1)
        result["raw_probability_high_performance"] = result[
            ["raw_probability_high_client_performer", "raw_probability_high_stakes_performer"]
        ].min(axis=1)
        result["classification_threshold"] = max(
            MIN_HIGH_PERFORMER_THRESHOLD,
            float(model_bundle.get("classification_threshold", MIN_HIGH_PERFORMER_THRESHOLD)),
        )
        result["predicted_high_performer"] = (
            result["predicted_high_client_performer"].eq(1)
            & result["predicted_high_stakes_performer"].eq(1)
        ).astype(int)

    result = _add_review_flags(result, model_bundle)

    # =========================================================
    # BUSINESS RECOMMENDATION
    # =========================================================

    result[
        "recommendation"
    ] = result.apply(
        assign_merchandising_recommendation,
        axis=1,
    )

    result[
        "scored_at"
    ] = pd.Timestamp.now(
        tz="UTC"
    )

    return result


def eligible_checkpoint_for_age(
    age: float,
    checkpoints: list[int],
    exact_checkpoint_only: bool = True,
):

    if pd.isna(age):
        return None

    age = int(age)

    if exact_checkpoint_only:
        return (
            age
            if age in checkpoints
            else None
        )

    eligible = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint <= age
    ]

    return (
        max(eligible)
        if eligible
        else None
    )


def run_daily_lifecycle_scoring(
    data,
    model_bundles,
    horizon_strategy,
    *,
    as_of_date=None,
    scoring_log=None,
    exact_checkpoint_only=True,
):

    as_of_date = pd.Timestamp(
        data["Date"].max()
        if as_of_date is None
        else as_of_date
    ).normalize()

    history = data[
        data["Date"] <= as_of_date
    ].copy()

    scoring_log = (
        pd.DataFrame()
        if scoring_log is None
        else scoring_log.copy()
    )

    status = (
        history.groupby("Game_ID", as_index=False)["First_Observed_Date"]
        .min()
        .assign(
            Current_Age=lambda frame: (
                as_of_date - pd.to_datetime(frame["First_Observed_Date"])
            ).dt.days + 1
        )
    )

    new_scores = []

    already = set()

    if (
        not scoring_log.empty
        and {
            "Game_ID",
            "Checkpoint",
            "Horizon",
        }.issubset(
            scoring_log.columns
        )
    ):

        already = set(
            zip(
                scoring_log["Game_ID"],
                scoring_log["Checkpoint"],
                scoring_log["Horizon"],
            )
        )

    configured = {
        (checkpoint, horizon)
        for checkpoint, horizons in horizon_strategy.items()
        for horizon in horizons
    }

    for checkpoint, horizon in sorted(model_bundles):
        if configured and (checkpoint, horizon) not in configured:
            continue

        model_bundle = model_bundles.get((checkpoint, horizon))
        if model_bundle is None:
            continue

        eligible = status[status["Current_Age"] >= checkpoint].copy()
        if exact_checkpoint_only:
            eligible = eligible[eligible["Current_Age"] >= checkpoint]

        eligible_ids = [
            game_id
            for game_id in eligible["Game_ID"].tolist()
            if (game_id, checkpoint, horizon) not in already
        ]

        if not eligible_ids:
            continue

        scoring_data = history[history["Game_ID"].isin(eligible_ids)]
        snapshot = build_lifecycle_snapshot(
            scoring_data,
            checkpoint,
            horizon,
            require_future=False,
            as_of_date=as_of_date,
        )

        scores = score_snapshot(snapshot, model_bundle)
        scores["Checkpoint"] = checkpoint
        scores["Horizon"] = horizon
        scores["As_Of_Date"] = as_of_date
        new_scores.append(scores)

    if not new_scores:
        return (
            pd.DataFrame(),
            scoring_log,
        )

    predictions = pd.concat(
        new_scores,
        ignore_index=True,
    )

    log_addition = predictions[
        [
            "Game_ID",
            "Checkpoint",
            "Horizon",
            "As_Of_Date",
            "scored_at",
        ]
    ]

    updated = pd.concat(
        [
            scoring_log,
            log_addition,
        ],
        ignore_index=True,
    ).drop_duplicates(
        [
            "Game_ID",
            "Checkpoint",
            "Horizon",
        ],
        keep="last",
    )

    return (
        predictions,
        updated,
    )
