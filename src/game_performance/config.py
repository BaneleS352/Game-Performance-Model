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
    rolling_validation_folds: int = 4
    final_test_fraction: float = 0.15
    min_training_rows: int = 100
    min_validation_rows: int = 25
    min_calibration_rows: int = 50
    calibration_bins: int = 10
    forecast_horizon: int = 7
    candidate_checkpoints: tuple[int, ...] = (21, 28, 35, 42, 49, 56)
    training_window_days: tuple[int | None, ...] = (None, 30, 60, 90, 120)
    selection_mode: str = "quick"
    representative_tuning_checkpoint: int = 21
    representative_tuning_horizon: int = 7

    @property
    def horizon_strategy(self) -> dict[int, list[int]]:
        return {checkpoint: [self.forecast_horizon] for checkpoint in self.candidate_checkpoints}

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
