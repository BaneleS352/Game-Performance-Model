import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
module = types.ModuleType("catboost")
module.CatBoostClassifier = type("CatBoostClassifier", (), {})
module.CatBoostRegressor = type("CatBoostRegressor", (), {})
sys.modules["catboost"] = module

from game_performance.data import clean_game_daily, load_raw_workbook
from game_performance.config import Settings
from game_performance.features import build_all_historical_snapshots
from game_performance.modeling import _baseline_prediction
from game_performance.validation import regression_metrics, split_development_and_holdout

raw = load_raw_workbook(Path("data/raw/Tournament Daily Dataset Feb26-Jul26.xlsx"))
data = clean_game_daily(raw)
settings = Settings(source_file=Path("data/raw/Tournament Daily Dataset Feb26-Jul26.xlsx"))
snapshots = build_all_historical_snapshots(data, settings.horizon_strategy)
results = {"raw_rows": len(raw), "games": int(data["Game_ID"].nunique()), "checkpoints": {}}

for (checkpoint, horizon), snapshot in snapshots.items():
    _, holdout, holdout_start = split_development_and_holdout(snapshot, settings.final_test_fraction)
    metrics = {}
    for target in ("target_clients", "target_stakes"):
        eligible = holdout.dropna(subset=[target])
        metrics[target] = regression_metrics(eligible[target], _baseline_prediction(eligible, target))
    results["checkpoints"][f"{checkpoint}->{checkpoint + horizon}"] = {
        "snapshot_rows": len(snapshot),
        "holdout_rows": len(holdout),
        "holdout_start": str(holdout_start.date()),
        "horizon": horizon,
        "baseline": metrics,
    }

Path("artifacts/accuracy_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
