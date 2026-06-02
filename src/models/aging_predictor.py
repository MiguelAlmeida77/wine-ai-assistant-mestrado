"""Aging predictor.

We train a GradientBoosting regressor on the synthetic aging trajectories so
the Streamlit app can query arbitrary (chemistry, year, wine_type) tuples
without re-running the analytical equations every time, and so the thesis can
show an ML model learning the joint behaviour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from ..data.loader import FEATURE_COLUMNS
from ..data.synthetic_aging import build_synthetic_dataset, project_chemistry, quality_curve
from .quality_predictor import MODELS_DIR

AGING_FEATURES = FEATURE_COLUMNS + ["year", "is_red"]


@dataclass
class AgingModel:
    estimator: object
    metrics: dict

    def predict_curve(
        self,
        initial: dict,
        wine_type: str,
        initial_quality: float,
        years_grid: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Return year-by-year prediction using the learned model.

        The analytical chemistry projection is still used to fill the per-year
        feature vector — the learned model contributes the quality response.
        """
        if years_grid is None:
            years_grid = np.arange(0.0, 15.5, 0.5)
        rows = []
        for t in years_grid:
            chem = project_chemistry(initial, float(t), wine_type)
            feat = {c: chem[c] for c in FEATURE_COLUMNS}
            feat["year"] = float(t)
            feat["is_red"] = 1 if wine_type == "red" else 0
            rows.append(feat)
        X = pd.DataFrame(rows)[AGING_FEATURES]
        pred = self.estimator.predict(X)
        # Anchor: at t=0 the prediction should equal initial_quality.
        # Shift the whole curve by the residual at t=0 so the curve passes
        # through the user's stated current quality.
        offset = initial_quality - pred[0]
        pred_shifted = pred + offset
        out = pd.DataFrame(rows)
        out["predicted_quality"] = pred_shifted
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "AgingModel":
        return joblib.load(path)


def train(seed: int = 42) -> AgingModel:
    df = build_synthetic_dataset()
    df["is_red"] = (df["type"] == "red").astype(int)
    X = df[AGING_FEATURES]
    y = df["predicted_quality"]
    groups = df["sample_id"].astype(str) + "_" + df["type"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    est = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        random_state=seed,
    )
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_groups_train": int(pd.Series(groups.iloc[train_idx]).nunique()),
        "n_groups_test": int(pd.Series(groups.iloc[test_idx]).nunique()),
    }
    model = AgingModel(estimator=est, metrics=metrics)
    model.save(MODELS_DIR / "aging.joblib")
    (MODELS_DIR / "aging_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return model


if __name__ == "__main__":
    model = train()
    print(json.dumps(model.metrics, indent=2))

    # Demo curve
    sample = {
        "fixed acidity": 8.0,
        "volatile acidity": 0.4,
        "citric acid": 0.3,
        "residual sugar": 2.5,
        "chlorides": 0.07,
        "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 110.0,
        "density": 0.997,
        "pH": 3.3,
        "sulphates": 0.65,
        "alcohol": 12.5,
    }
    curve = model.predict_curve(sample, "red", initial_quality=6.0)
    print(curve[["year", "predicted_quality", "volatile acidity", "free sulfur dioxide"]].iloc[::4])
