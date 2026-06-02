"""Market acceptance predictor.

Maps wine metadata + textual tasting note → critic score (80–100).

This is intentionally orthogonal to `quality_predictor`: it does NOT use
chemistry (the Wine Reviews dataset has none). Instead it learns the
*market lens* — what kind of style/region/price wins critic points.

The two models are combined at decision time into a composite score, giving
the enologist both a technical estimate (from chemistry) and a market
estimate (from style/positioning).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from ..data.reviews_loader import load_reviews
from .quality_predictor import MODELS_DIR

# Cardinality caps — categories beyond top-N collapse to 'other'
TOP_N = {"country": 25, "variety": 40, "province": 50, "taster_name": 20}
NUM_COLS = ["log_price"]
CAT_COLS = ["country", "variety", "province", "taster_name"]
TEXT_COL = "description"


def _cap_categories(df: pd.DataFrame, col: str, top: int) -> pd.Series:
    s = df[col].astype("object").where(df[col].notna(), other="unknown")
    keep = set(s.value_counts().head(top).index)
    return s.where(s.isin(keep), other="other")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_price"] = np.log1p(df["price"].fillna(df["price"].median()))
    for c, n in TOP_N.items():
        out[c] = _cap_categories(df, c, n)
    out[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return out


@dataclass
class MarketModel:
    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)

    def predict(self, sample: dict | pd.DataFrame) -> np.ndarray:
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        # Ensure all needed columns exist
        for c in CAT_COLS:
            if c not in sample.columns:
                sample[c] = "unknown"
        if "price" in sample.columns and "log_price" not in sample.columns:
            sample = sample.copy()
            sample["log_price"] = np.log1p(sample["price"].fillna(sample["price"].median() if len(sample) > 0 else 30))
        if "log_price" not in sample.columns:
            sample["log_price"] = np.log1p(30.0)
        if TEXT_COL not in sample.columns:
            sample = sample.copy()
            sample[TEXT_COL] = ""
        return self.pipeline.predict(sample)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "MarketModel":
        return joblib.load(path)


def _load_best_market_params() -> dict:
    path = MODELS_DIR / "best_params_market.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("best_params", {})


def _build_pipeline(override_params: dict | None = None) -> Pipeline:
    # Cap text feature count to keep model size reasonable
    text_pipe = TfidfVectorizer(
        max_features=400,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=20,
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CAT_COLS,
            ),
            ("text", text_pipe, TEXT_COL),
        ]
    )
    # XGB tuned for tabular+sparse text
    xgb_params = dict(
        n_estimators=700, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0,
    )
    if override_params:
        xgb_params.update(override_params)
    est = XGBRegressor(
        **xgb_params, random_state=42, n_jobs=-1, tree_method="hist",
    )
    return Pipeline([("pre", pre), ("est", est)])


def train(seed: int = 42, sample: int | None = None) -> MarketModel:
    df = load_reviews()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
    feats = prepare_features(df)
    y = df["points"].astype(float)
    groups = df["winery"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(feats, y, groups=groups))
    X_train, X_test = feats.iloc[train_idx], feats.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = _build_pipeline(override_params=_load_best_market_params() or None)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_wineries_train": int(df.iloc[train_idx]["winery"].nunique()),
        "n_wineries_test": int(df.iloc[test_idx]["winery"].nunique()),
    }
    model = MarketModel(pipeline=pipe, metrics=metrics)
    model.save(MODELS_DIR / "market.joblib")
    (MODELS_DIR / "market_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return model


def composite_score(
    quality_0_10: float, market_0_100: float, weight_market: float = 0.4
) -> float:
    """Combine quality (0–10) with market (80–100) into a 0–10 composite.

    Market is rescaled from [80, 100] → [0, 10] so the two are commensurate.
    """
    market_rescaled = (np.clip(market_0_100, 80, 100) - 80) / 2.0
    return float((1 - weight_market) * quality_0_10 + weight_market * market_rescaled)


if __name__ == "__main__":
    model = train()
    print(json.dumps(model.metrics, indent=2))
