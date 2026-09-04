import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_performance.config import Settings
from game_performance.dashboard import export_dashboard
from game_performance.io import load_artifact, load_dataframe, save_artifact, save_dataframe
from game_performance.modeling import train_all_model_families
from game_performance.reporting import build_holdout_comparison, export_excel
from game_performance.scoring import run_daily_lifecycle_scoring

settings = Settings(source_file=Path("data/raw/Tournament Daily Dataset Feb26-Jul26.xlsx"))
settings.ensure_directories()

data = load_dataframe(settings.processed_dir / "clean_game_daily.pkl")
snapshots = load_artifact(settings.feature_dir / "historical_snapshots.joblib")

results = train_all_model_families(snapshots, settings, mode="standard")
save_artifact(results, settings.model_dir / "training_results.joblib")

forecasts, scoring_log = run_daily_lifecycle_scoring(
    data,
    results["model_bundles"],
    settings.horizon_strategy,
    as_of_date=data["Date"].max(),
    exact_checkpoint_only=False,
)
save_dataframe(forecasts, settings.report_dir / "current_predictions.pkl")
save_dataframe(scoring_log, settings.report_dir / "scoring_log.pkl")

comparison = build_holdout_comparison(results["holdout_predictions"])
report = export_excel(
    settings.report_dir / "model_business_report.xlsx",
    comparison=comparison,
    forecasts=forecasts,
    regression=results["regression_evaluation"],
    classification=results["classification_evaluation"],
    calibration=results["classification_calibration"],
)
dashboard = export_dashboard()

print(f"report={report}")
print(f"dashboard={dashboard}")
print(results["regression_evaluation"].to_string(index=False))
