"""Uncertainty quantification for the quality predictor.

Two complementary approaches:

1. **Quantile XGBoost** — three XGB regressors trained with quantile loss at
   α=0.1, 0.5, 0.9. Cheap, model-aware, but the coverage is only as good as
   the model's calibration on the underlying data.

2. **Split-conformal prediction** — distribution-free: fit on a train split,
   compute absolute residuals on a held-out *calibration* split, and form
   prediction intervals as median ± q_{1-α}(|residuals|). Gives a
   mathematically guaranteed marginal coverage of (1 − α) on exchangeable
   test points, independent of the base model.

The Streamlit UI shows both so the enologist sees a coherent interval *and*
a distribution-free safety net. The thesis can report empirical coverage
for both methods on a held-out fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from ..data.loader import FEATURE_COLUMNS
from ..data.preprocessor import add_engineered_features


def _xgb(alpha: float, seed: int = 42, **kwargs) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=kwargs.pop("n_estimators", 600),
        max_depth=kwargs.pop("max_depth", 6),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        subsample=kwargs.pop("subsample", 0.9),
        colsample_bytree=kwargs.pop("colsample_bytree", 0.9),
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **kwargs,
    )


@dataclass
class IntervalModel:
    """Bundle of quantile estimators + conformal calibration."""

    wine_type: str
    feature_cols: list[str]
    lo: XGBRegressor
    mid: XGBRegressor
    hi: XGBRegressor
    conformal_q: float = 0.0          # absolute residual quantile for 80% CI
    alpha: float = 0.2                # nominal miscoverage
    metrics: dict = field(default_factory=dict)

    def _features(self, sample: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        sample = add_engineered_features(sample)
        return sample[self.feature_cols]

    def predict(self, sample: dict | pd.DataFrame) -> dict:
        X = self._features(sample)
        median = self.mid.predict(X)
        lo = self.lo.predict(X)
        hi = self.hi.predict(X)
        # Ensure ordering after independent quantile fits (occasional crossing)
        lo = np.minimum(lo, median)
        hi = np.maximum(hi, median)
        conformal_lo = median - self.conformal_q
        conformal_hi = median + self.conformal_q
        return {
            "median": median,
            "quantile_lo": lo,
            "quantile_hi": hi,
            "conformal_lo": conformal_lo,
            "conformal_hi": conformal_hi,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "IntervalModel":
        return joblib.load(path)


def train_interval(
    df_with_type: pd.DataFrame,
    wine_type: str,
    alpha: float = 0.2,
    seed: int = 42,
) -> IntervalModel:
    """Train quantile XGB + calibrate split-conformal residuals.

    Splits: 70% train, 15% calibration, 15% test. The conformal residual
    quantile is computed on the calibration split.
    """
    from .quality_predictor import basic_clean  # local import to avoid cycle

    sub = df_with_type[df_with_type["type"] == wine_type].copy()
    sub = basic_clean(sub)
    sub = add_engineered_features(sub)
    y = sub["quality"].values
    feature_cols = [c for c in sub.columns if c not in {"quality", "type"}]
    X = sub[feature_cols]

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed
    )

    lo = _xgb(alpha / 2, seed=seed).fit(X_tr, y_tr)
    mid = _xgb(0.5, seed=seed).fit(X_tr, y_tr)
    hi = _xgb(1 - alpha / 2, seed=seed).fit(X_tr, y_tr)

    # Split-conformal: residuals on calibration set
    pred_cal = mid.predict(X_cal)
    residuals = np.abs(y_cal - pred_cal)
    # 1 - alpha quantile, with finite-sample correction
    n = len(residuals)
    q_idx = int(np.ceil((n + 1) * (1 - alpha)))
    q_idx = min(q_idx, n) - 1
    conformal_q = float(np.sort(residuals)[q_idx])

    # Test-set evaluation: coverage and width for both intervals
    p_te = mid.predict(X_te)
    lo_te = np.minimum(lo.predict(X_te), p_te)
    hi_te = np.maximum(hi.predict(X_te), p_te)
    quant_coverage = float(np.mean((y_te >= lo_te) & (y_te <= hi_te)))
    quant_width = float(np.mean(hi_te - lo_te))
    conf_lo_te = p_te - conformal_q
    conf_hi_te = p_te + conformal_q
    conf_coverage = float(np.mean((y_te >= conf_lo_te) & (y_te <= conf_hi_te)))
    conf_width = float(2 * conformal_q)

    metrics = {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "quantile_xgb": {"empirical_coverage": quant_coverage, "mean_width": quant_width},
        "conformal": {"empirical_coverage": conf_coverage, "mean_width": conf_width},
        "n_train": int(len(X_tr)),
        "n_calibration": int(len(X_cal)),
        "n_test": int(len(X_te)),
        "median_test_rmse": float(np.sqrt(np.mean((y_te - p_te) ** 2))),
    }

    return IntervalModel(
        wine_type=wine_type,
        feature_cols=feature_cols,
        lo=lo,
        mid=mid,
        hi=hi,
        conformal_q=conformal_q,
        alpha=alpha,
        metrics=metrics,
    )
