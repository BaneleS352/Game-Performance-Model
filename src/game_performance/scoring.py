from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    PREDICTION_WEEK_END,
    REGRESSION_TARGETS,
    build_lifecycle_snapshot,
    build_week4_snapshot,
    prepare_features,
)

from .modeling import (
    _baseline_prediction,
    _prediction_from_fit,
    apply_probability_calibration,
    confidence_from_prediction,
    prediction_interval_from_table,
)


def assign_merchandising_recommendation(
    row: pd.Series,
) -> str:

    probability = row.get(
        "probability_high_performance",
        np.nan,
    )

    predicted_clients = row.get(
        "predicted_clients_week_4",
        np.nan,
    )

    predicted_stakes = row.get(
        "predicted_stakes_week_4",
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
        return "Priority placement for week 4"

    if (
        pd.notna(probability)
        and probability >= 0.60
        and pd.notna(predicted_stakes)
        and predicted_stakes > 0
    ):
        return "Increase visibility for week 4"

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
        "Obs_Window",
        "Horizon",
        "active_observation_days",
        "observation_coverage",
    ]

    result = snapshot[cols].copy()

    # =========================================================
    # REGRESSION FORECASTS
    # =========================================================

    for target, model in model_bundle[
        "regressors"
    ].items():

        if target not in REGRESSION_TARGETS:
            continue

        short = target.replace(
            "target_",
            "",
        )

        transform = (
            model_bundle
            .get(
                "target_transforms",
                {},
            )
            .get(target)
            or model_bundle
            .get(
                "regression_target_transforms",
                {},
            )
            .get(
                target,
                "raw",
            )
        )

        # -----------------------------------------------------
        # Model prediction
        # -----------------------------------------------------

        model_pred = _prediction_from_fit(
            target,
            model.predict(x),
            transform,
        )

        # -----------------------------------------------------
        # Baseline prediction
        # -----------------------------------------------------

        baseline_pred = _baseline_prediction(
            snapshot,
            target,
        )

        # -----------------------------------------------------
        # Blend model + baseline
        # -----------------------------------------------------

        blend_weight = (
            model_bundle
            .get(
                "blend_weights",
                {},
            )
            .get(
                target,
                1.0,
            )
        )

        pred = (
            blend_weight * model_pred
            + (1 - blend_weight)
            * baseline_pred
        )

        result[
            f"predicted_{short}_week_4"
        ] = pred

        prediction_low, prediction_high = prediction_interval_from_table(
            pred,
            model_bundle
            .get(
                "prediction_interval_tables",
                {},
            )
            .get(target),
        )

        result[
            f"predicted_{short}_week_4_low"
        ] = prediction_low

        result[
            f"predicted_{short}_week_4_high"
        ] = prediction_high

        # =====================================================
        # GAME-SPECIFIC CONFIDENCE
        # =====================================================

        confidence_table = (
            model_bundle
            .get(
                "confidence_tables",
                {},
            )
            .get(target)
        )

        metric_confidence = (
            model_bundle
            .get(
                "confidence_scores",
                {},
            )
            .get(target)
        )

        # If an older model bundle does not contain a proper
        # confidence table, construct a fallback-only table.
        if not confidence_table:

            confidence_table = {
                "fallback": metric_confidence,
                "bins": [],
            }

        elif (
            pd.isna(
                confidence_table.get(
                    "fallback",
                    np.nan,
                )
            )
            and metric_confidence is not None
        ):

            confidence_table = (
                confidence_table.copy()
            )

            confidence_table[
                "fallback"
            ] = metric_confidence

        # IMPORTANT:
        #
        # Always call confidence_from_prediction().
        #
        # Do NOT directly assign metric_confidence to every row.
        result[
            f"confidence_{short}"
        ] = confidence_from_prediction(
            pred,
            confidence_table,
            snapshot.get(
                "observation_coverage"
            ),
        )

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
            .get(goal, model_bundle.get("classification_threshold", 0.50))
        )

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
        result["classification_threshold"] = float(model_bundle.get("classification_threshold", 0.50))
        result["predicted_high_performer"] = (
            result["predicted_high_client_performer"].eq(1)
            & result["predicted_high_stakes_performer"].eq(1)
        ).astype(int)

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
        history
        .groupby(
            "Game_ID",
            as_index=False,
        )
        .agg(
            Current_Age=(
                "Days_Since_First_Observed",
                "max",
            )
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
        if checkpoint + horizon == PREDICTION_WEEK_END:
            snapshot = build_week4_snapshot(
                scoring_data,
                checkpoint,
                require_future=False,
            )
        else:
            snapshot = build_lifecycle_snapshot(
                scoring_data,
                checkpoint,
                horizon,
                require_future=False,
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
