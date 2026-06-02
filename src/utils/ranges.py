"""Sensible slider ranges derived from the UCI dataset for the Streamlit UI."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..data.loader import FEATURE_COLUMNS, load_raw


@lru_cache(maxsize=2)
def feature_ranges(wine_type: str) -> dict[str, tuple[float, float, float]]:
    """Return (min, max, median) per feature for the given wine type."""
    df = load_raw(wine_type if wine_type in ("red", "white") else "both")
    out: dict[str, tuple[float, float, float]] = {}
    for c in FEATURE_COLUMNS:
        s = df[c].dropna()
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        med = float(s.median())
        out[c] = (lo, hi, med)
    return out


def default_chemistry(wine_type: str) -> dict[str, float]:
    return {c: v[2] for c, v in feature_ranges(wine_type).items()}
