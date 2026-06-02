"""Cleaning and feature engineering for UCI Wine Quality data."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .loader import FEATURE_COLUMNS


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and rows with missing values in core columns."""
    cols = FEATURE_COLUMNS + ["quality"]
    if "type" in df.columns:
        cols = cols + ["type"]
    df = df.dropna(subset=cols).copy()
    df = df.drop_duplicates(subset=cols).reset_index(drop=True)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features known to correlate with perceived quality."""
    df = df.copy()
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
    # SO2 bound = total - free (proxy for already-reacted preservative)
    df["bound_sulfur_dioxide"] = (
        df["total sulfur dioxide"] - df["free sulfur dioxide"]
    ).clip(lower=0)
    # Alcohol-to-sugar ratio — body proxy
    df["alcohol_sugar_ratio"] = df["alcohol"] / (df["residual sugar"] + 1e-3)
    return df


def split_xy(
    df: pd.DataFrame, target: str = "quality", test_size: float = 0.2, seed: int = 42
):
    feature_cols = [c for c in df.columns if c not in {target, "type"}]
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols
