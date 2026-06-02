"""Blend recommender.

Given a set of base wines (each with a chemical fingerprint), find the
proportions that maximise predicted quality — optionally subject to user-defined
constraints on the resulting blend's chemistry (e.g. minimum alcohol, maximum
volatile acidity) and a target style vector.

The mixing model assumes linear combination of chemical features by volume
fraction, which is a standard first-order approximation in enology — it is
exact for additive properties (alcohol on a mass basis is close enough, sugar,
SO₂) and approximate for non-additive ones (pH, density). Documenting this in
the thesis is important.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.loader import FEATURE_COLUMNS
from .quality_predictor import MODELS_DIR, QualityModel


@dataclass
class BaseWine:
    name: str
    chemistry: dict  # keys = FEATURE_COLUMNS

    def vector(self) -> np.ndarray:
        return np.array([self.chemistry[c] for c in FEATURE_COLUMNS], dtype=float)


@dataclass
class BlendConstraints:
    min_alcohol: float | None = None
    max_alcohol: float | None = None
    max_volatile_acidity: float | None = None
    max_total_so2: float | None = None
    target_profile: dict | None = None  # FEATURE_COLUMNS -> target value
    target_weight: float = 0.0  # 0 = ignore profile, >0 = penalise distance


@dataclass
class BlendResult:
    proportions: dict
    blended_chemistry: dict
    predicted_quality: float
    converged: bool
    message: str

    def to_frame(self) -> pd.DataFrame:
        prop = pd.Series(self.proportions, name="proporção")
        return prop.to_frame().assign(percentagem=lambda d: (d["proporção"] * 100).round(2))


def mix_chemistry(base_wines: Sequence[BaseWine], proportions: np.ndarray) -> dict:
    """Linear (volume-fraction) mix of chemical vectors."""
    matrix = np.stack([w.vector() for w in base_wines])  # (n_wines, n_features)
    mixed = (matrix * proportions[:, None]).sum(axis=0)
    return {c: float(v) for c, v in zip(FEATURE_COLUMNS, mixed)}


def recommend_blend(
    base_wines: Sequence[BaseWine],
    quality_model: QualityModel,
    constraints: BlendConstraints | None = None,
    seed: int = 42,
) -> BlendResult:
    if len(base_wines) < 2:
        raise ValueError("Need at least 2 base wines to blend.")
    constraints = constraints or BlendConstraints()
    n = len(base_wines)

    target_vec = None
    if constraints.target_profile and constraints.target_weight > 0:
        target_vec = np.array(
            [constraints.target_profile.get(c, np.nan) for c in FEATURE_COLUMNS]
        )

    def objective(p: np.ndarray) -> float:
        chem = mix_chemistry(base_wines, p)
        q = float(quality_model.predict(chem)[0])
        loss = -q
        if target_vec is not None:
            mixed_vec = np.array([chem[c] for c in FEATURE_COLUMNS])
            mask = ~np.isnan(target_vec)
            if mask.any():
                # Normalised L2 distance on the dimensions the user specified
                diff = (mixed_vec[mask] - target_vec[mask]) / (
                    np.abs(target_vec[mask]) + 1e-3
                )
                loss += constraints.target_weight * float(np.mean(diff**2))
        return loss

    # Constraints: proportions sum to 1, each in [0, 1]
    cons: list[dict] = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]

    def make_le(idx_feature: int, upper: float):
        # upper - mixed_value >= 0
        return lambda p: upper - mix_chemistry(base_wines, p)[FEATURE_COLUMNS[idx_feature]]

    def make_ge(idx_feature: int, lower: float):
        return lambda p: mix_chemistry(base_wines, p)[FEATURE_COLUMNS[idx_feature]] - lower

    idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    if constraints.min_alcohol is not None:
        cons.append({"type": "ineq", "fun": make_ge(idx["alcohol"], constraints.min_alcohol)})
    if constraints.max_alcohol is not None:
        cons.append({"type": "ineq", "fun": make_le(idx["alcohol"], constraints.max_alcohol)})
    if constraints.max_volatile_acidity is not None:
        cons.append({"type": "ineq", "fun": make_le(idx["volatile acidity"], constraints.max_volatile_acidity)})
    if constraints.max_total_so2 is not None:
        cons.append({"type": "ineq", "fun": make_le(idx["total sulfur dioxide"], constraints.max_total_so2)})

    bounds = [(0.0, 1.0)] * n

    # Multi-start: try a couple of initialisations because SLSQP is local.
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)]
    for _ in range(4):
        p = rng.dirichlet(np.ones(n))
        starts.append(p)

    best: BlendResult | None = None
    best_loss = np.inf
    last_message = ""
    for start in starts:
        res = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-6},
        )
        last_message = res.message
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            p = np.clip(res.x, 0.0, 1.0)
            p = p / p.sum()
            chem = mix_chemistry(base_wines, p)
            q = float(quality_model.predict(chem)[0])
            best = BlendResult(
                proportions={w.name: float(pi) for w, pi in zip(base_wines, p)},
                blended_chemistry=chem,
                predicted_quality=q,
                converged=True,
                message=res.message,
            )

    if best is None:
        # Fallback: return equal mix
        p = np.full(n, 1.0 / n)
        chem = mix_chemistry(base_wines, p)
        q = float(quality_model.predict(chem)[0])
        best = BlendResult(
            proportions={w.name: float(pi) for w, pi in zip(base_wines, p)},
            blended_chemistry=chem,
            predicted_quality=q,
            converged=False,
            message=f"Optimization failed: {last_message}. Returned equal mix.",
        )
    return best


def sample_base_wines_from_dataset(
    df: pd.DataFrame, wine_type: str, n: int = 5, seed: int = 7
) -> list[BaseWine]:
    """Pick `n` distinct wines from the dataset to use as base wines.

    Useful for the demo UI: the user can click "exemplo" to populate a realistic
    set of bases without having to type 11 numbers by hand.
    """
    rng = np.random.default_rng(seed)
    subset = df[df["type"] == wine_type].reset_index(drop=True)
    idx = rng.choice(len(subset), size=n, replace=False)
    bases = []
    for i, k in enumerate(idx):
        row = subset.iloc[k]
        bases.append(
            BaseWine(
                name=f"Base {i+1} (q={int(row['quality'])})",
                chemistry={c: float(row[c]) for c in FEATURE_COLUMNS},
            )
        )
    return bases


if __name__ == "__main__":
    from ..data.loader import load_raw

    df = load_raw("both")
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    bases = sample_base_wines_from_dataset(df, "red", n=4)
    result = recommend_blend(
        bases,
        model,
        constraints=BlendConstraints(min_alcohol=10.5, max_volatile_acidity=0.7),
    )
    print("Proporções:")
    for name, p in result.proportions.items():
        print(f"  {name}: {p*100:.1f}%")
    print(f"\nQualidade prevista do blend: {result.predicted_quality:.2f}")
    print(f"Convergiu: {result.converged}")
