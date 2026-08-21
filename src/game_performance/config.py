from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    source_file: Path = field(default_factory=lambda: Path(os.getenv("GAME_DATA_FILE", "data/raw/game_performance.xlsx")))
    random_seed: int = 42
    high_performer_percentile: float = 0.80
    rolling_validation_folds: int = 2
    final_test_fraction: float = 0.15
    min_training_rows: int = 100
    min_validation_rows: int = 25
    representative_tuning_checkpoint: int = 14
    representative_tuning_horizon: int = 30

    horizon_strategy: dict[int, list[int]] = field(default_factory=lambda: {
        7: [7, 14, 30],
        14: [7, 14, 30, 60],
        21: [14, 30, 60],
    })

    def __post_init__(self) -> None:
        if not self.source_file.is_absolute():
            object.__setattr__(self, "source_file", self.project_root / self.source_file)

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "artifacts" / "processed"

    @property
    def feature_dir(self) -> Path:
        return self.project_root / "artifacts" / "features"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "artifacts" / "models"

    @property
    def report_dir(self) -> Path:
        return self.project_root / "artifacts" / "reports"

    def ensure_directories(self) -> None:
        for path in (self.processed_dir, self.feature_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
