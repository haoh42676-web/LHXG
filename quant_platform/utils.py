from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def to_number(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace("None", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clamp(value: float, low: float, high: float) -> float:
    if value is None or math.isnan(value):
        return low
    return max(low, min(high, value))


def zscore(values: Iterable[float]) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std
