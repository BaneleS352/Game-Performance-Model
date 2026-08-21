from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(Path(path))


def save_artifact(obj: Any, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(path: str | Path) -> Any:
    return joblib.load(Path(path))
