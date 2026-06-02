"""Counterfactual explanations for the quality predictor.

Given a wine with current chemistry `x` and predicted quality `q(x)`, find
the *minimum* chemical perturbation `Δ` such that `q(x + Δ) ≥ target`. The
cost of changing each feature is weighted by its standard deviation in the
training set, so the optimiser favours small relative changes.

Why this matters for the enologist:
  - SHAP tells you *which features mattered* for the current prediction.
  - Counterfactuals tell you *what to change* to reach a target quality.
  - The two are complementary: SHAP explains, counterfactuals act.

Output is a ranked list of suggested edits (top-K), each with the magnitude
of the change and whether the new value is still within plausible enological
bounds (taken from the 1st–99th percentile of the UCI dataset).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.loader import FEATURE_COLUMNS, FEATURE_LABELS_PT
from ..utils.ranges import feature_ranges
from .quality_predictor import QualityModel


# Fast path: build the full feature vector (including engineered features) from
# a numpy array of the 11 raw chemistry features, in the order expected by the
# trained estimator. Avoids the per-call pandas overhead that made the SLSQP
# loop unusable.
def _build_feature_matrix(
    raw_x: np.ndarray, feature_cols: list[str]
) -> np.ndarray:
    """Return a 2-D (1, n_features) array aligned with `feature_cols`.

    Supports the engineered features added by `preprocessor.add_engineered_features`:
    `total_acidity`, `bound_sulfur_dioxide`, `alcohol_sugar_ratio`.
    """
    raw_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    n = len(feature_cols)
    out = np.empty(n, dtype=float)
    for j, c in enumerate(feature_cols):
        if c in raw_idx:
            out[j] = raw_x[raw_idx[c]]
        elif c == "total_acidity":
            out[j] = raw_x[raw_idx["fixed acidity"]] + raw_x[raw_idx["volatile acidity"]]
        elif c == "bound_sulfur_dioxide":
            out[j] = max(
                raw_x[raw_idx["total sulfur dioxide"]]
                - raw_x[raw_idx["free sulfur dioxide"]],
                0.0,
            )
        elif c == "alcohol_sugar_ratio":
            out[j] = raw_x[raw_idx["alcohol"]] / (raw_x[raw_idx["residual sugar"]] + 1e-3)
        else:
            raise ValueError(f"Unknown feature column: {c}")
    return out.reshape(1, -1)

# Features the enologist can actually adjust in the cellar (we exclude
# `density` and `pH` from suggestions because they are emergent, not
# directly dosable). They are still allowed to vary in the optimisation,
# they just don't appear in the actionable changes list.
ACTIONABLE = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "sulphates",
    "alcohol",
}

# Cost weights: 1 / std (a 1-std change costs 1 unit). Density/pH get high
# cost so the optimiser rarely touches them.
NON_ACTIONABLE_PENALTY = 10.0


@dataclass
class Change:
    feature: str
    label_pt: str
    delta: float
    from_value: float
    to_value: float
    within_dataset_bounds: bool


@dataclass
class CounterfactualResult:
    target_quality: float
    achieved_quality: float
    initial_quality: float
    feasible: bool
    changes: list[Change]
    n_changes: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "variável": c.label_pt,
                    "actual": round(c.from_value, 3),
                    "proposta": round(c.to_value, 3),
                    "Δ": round(c.delta, 3),
                    "dentro da gama UCI": c.within_dataset_bounds,
                }
                for c in self.changes
            ]
        )


def find_counterfactual(
    current: dict,
    target_quality: float,
    quality_model: QualityModel,
    wine_type: str,
    top_k: int = 4,
    min_delta_per_feature: float = 0.05,
) -> CounterfactualResult:
    """Find minimum-cost edits to reach `target_quality`.

    Parameters
    ----------
    current : dict
        Raw chemistry (11 features).
    target_quality : float
        Desired quality on the 0–10 scale.
    quality_model : QualityModel
        Trained model used as the differentiable proxy (via scipy).
    wine_type : str
        "red" or "white" — used to pull dataset bounds.
    top_k : int
        Maximum number of features to report as actionable changes.
    min_delta_per_feature : float
        Changes below this (in original units, scaled by std) are dropped
        from the suggestions list.
    """
    ranges = feature_ranges(wine_type)
    x0 = np.array([current[c] for c in FEATURE_COLUMNS], dtype=float)

    # Std proxy from dataset range (99-1 percentile / ~4σ for normal)
    stds = np.array([
        max((ranges[c][1] - ranges[c][0]) / 4.0, 1e-3)
        for c in FEATURE_COLUMNS
    ])

    # Per-feature cost: low for actionable, high otherwise
    costs = np.array([
        1.0 if c in ACTIONABLE else NON_ACTIONABLE_PENALTY
        for c in FEATURE_COLUMNS
    ])

    estimator = quality_model.estimator
    feat_cols = quality_model.feature_cols

    def quality_of(x: np.ndarray) -> float:
        X = _build_feature_matrix(x, feat_cols)
        return float(estimator.predict(X)[0])

    initial_q = quality_of(x0)

    def cost(delta: np.ndarray) -> float:
        return float(np.sum(costs * (delta / stds) ** 2))

    def constraint(delta: np.ndarray) -> float:
        return quality_of(x0 + delta) - target_quality

    # Bounds: stay within dataset 1–99 percentile range
    bounds = []
    for c, x0i in zip(FEATURE_COLUMNS, x0):
        lo, hi, _ = ranges[c]
        bounds.append((lo - x0i, hi - x0i))

    # Multi-start to escape local minima — keep small for speed
    best: tuple[float, np.ndarray] | None = None
    rng = np.random.default_rng(7)
    starts = [np.zeros_like(x0)]
    for _ in range(3):
        starts.append(rng.normal(0, 0.1 * stds, size=x0.shape))

    for start in starts:
        res = minimize(
            cost,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"maxiter": 60, "ftol": 1e-4},
        )
        if not res.success:
            continue
        achieved = quality_of(x0 + res.x)
        if achieved >= target_quality - 0.05:
            c_val = cost(res.x)
            if best is None or c_val < best[0]:
                best = (c_val, res.x)

    if best is None:
        return CounterfactualResult(
            target_quality=target_quality,
            achieved_quality=initial_q,
            initial_quality=initial_q,
            feasible=False,
            changes=[],
            n_changes=0,
        )

    _, delta = best
    achieved = quality_of(x0 + delta)
    new_x = x0 + delta

    # Build ranked change list, restricted to actionable features and
    # non-trivial deltas.
    candidates: list[Change] = []
    for i, c in enumerate(FEATURE_COLUMNS):
        if c not in ACTIONABLE:
            continue
        d = float(delta[i])
        # Threshold in original units, normalised by std
        if abs(d / stds[i]) < min_delta_per_feature:
            continue
        lo, hi, _ = ranges[c]
        within = lo <= new_x[i] <= hi
        candidates.append(
            Change(
                feature=c,
                label_pt=FEATURE_LABELS_PT[c],
                delta=d,
                from_value=float(x0[i]),
                to_value=float(new_x[i]),
                within_dataset_bounds=within,
            )
        )

    # Sort by absolute magnitude in std units (most impactful first)
    candidates.sort(
        key=lambda c: abs(c.delta / stds[FEATURE_COLUMNS.index(c.feature)]),
        reverse=True,
    )
    top = candidates[:top_k]

    return CounterfactualResult(
        target_quality=target_quality,
        achieved_quality=achieved,
        initial_quality=initial_q,
        feasible=True,
        changes=top,
        n_changes=len(top),
    )


if __name__ == "__main__":
    from .quality_predictor import MODELS_DIR

    sample = {
        "fixed acidity": 8.0, "volatile acidity": 0.6, "citric acid": 0.2,
        "residual sugar": 2.5, "chlorides": 0.08, "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 60.0, "density": 0.997, "pH": 3.3,
        "sulphates": 0.5, "alcohol": 10.5,
    }
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    result = find_counterfactual(sample, target_quality=7.0,
                                  quality_model=model, wine_type="red")
    print(f"Inicial: {result.initial_quality:.2f}, alvo {result.target_quality}, "
          f"alcançado {result.achieved_quality:.2f}, viável={result.feasible}")
    print(result.to_frame().to_string(index=False))
