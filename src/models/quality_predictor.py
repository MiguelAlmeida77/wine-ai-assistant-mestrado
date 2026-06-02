"""Quality prediction model: chemistry → sensory score (0–10).

Trains a separate XGBoost regressor per wine type (red, white). Each model is
wrapped in a small class that handles feature ordering and persistence so the
Streamlit app can predict from a dict without worrying about column order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from ..data.loader import FEATURE_COLUMNS, PROCESSED_DIR, load_raw
from ..data.preprocessor import add_engineered_features, basic_clean, split_xy

MODELS_DIR = PROCESSED_DIR.parent.parent / "models"


@dataclass
class QualityModel:
    """Trained quality predictor for a single wine type."""

    wine_type: str
    feature_cols: list[str]
    estimator: object
    metrics: dict = field(default_factory=dict)

    def predict(self, sample: dict | pd.DataFrame) -> float | np.ndarray:
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        # Re-derive engineered features so callers only need raw chemistry
        sample = add_engineered_features(sample)
        X = sample[self.feature_cols]
        return self.estimator.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "QualityModel":
        return joblib.load(path)


def _evaluate(estimator, X_test, y_test) -> dict:
    pred = estimator.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def _load_best_params(wine_type: str) -> dict:
    """Load Optuna-tuned best params if they exist, else return empty dict."""
    path = MODELS_DIR / f"best_params_quality_{wine_type}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def train_for_type(df: pd.DataFrame, wine_type: str, seed: int = 42) -> QualityModel:
    subset = df[df["type"] == wine_type].copy()
    subset = basic_clean(subset)
    subset = add_engineered_features(subset)
    X_train, X_test, y_train, y_test, feature_cols = split_xy(subset, seed=seed)

    tuned = _load_best_params(wine_type)

    xgb_params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    )
    if tuned.get("xgboost", {}).get("best_params"):
        xgb_params.update(tuned["xgboost"]["best_params"])
    xgb = XGBRegressor(
        **xgb_params,
        random_state=seed, n_jobs=-1, tree_method="hist",
        objective="reg:squarederror",
    )
    xgb.fit(X_train, y_train)

    rf_params = dict(n_estimators=400, max_depth=None)
    if tuned.get("random_forest", {}).get("best_params"):
        rf_params = dict(tuned["random_forest"]["best_params"])
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **rf_params)
    rf.fit(X_train, y_train)

    xgb_metrics = _evaluate(xgb, X_test, y_test)
    rf_metrics = _evaluate(rf, X_test, y_test)

    # 5-fold CV on training portion for a robust baseline
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_r2 = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

    best_est = xgb if xgb_metrics["rmse"] <= rf_metrics["rmse"] else rf
    best_name = "xgboost" if best_est is xgb else "random_forest"

    metrics = {
        "winner": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
        "xgb_cv_r2_mean": float(cv_r2.mean()),
        "xgb_cv_r2_std": float(cv_r2.std()),
    }

    return QualityModel(
        wine_type=wine_type,
        feature_cols=feature_cols,
        estimator=best_est,
        metrics=metrics,
    )


def train_all(seed: int = 42) -> dict[str, QualityModel]:
    df = load_raw("both")
    models: dict[str, QualityModel] = {}
    summary = {}
    for wt in ("red", "white"):
        model = train_for_type(df, wt, seed=seed)
        model.save(MODELS_DIR / f"quality_{wt}.joblib")
        models[wt] = model
        summary[wt] = model.metrics
    (MODELS_DIR / "quality_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return models


if __name__ == "__main__":
    models = train_all()
    for wt, m in models.items():
        print(f"\n=== {wt.upper()} ===")
        print(json.dumps(m.metrics, indent=2))
