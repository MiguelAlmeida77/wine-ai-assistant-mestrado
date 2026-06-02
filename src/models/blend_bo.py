"""Bayesian / multi-objective blend optimization.

Complementary to the SLSQP recommender. Two modes:

1. **Single-objective BO** (Optuna TPE) — better global exploration of the
   non-convex quality surface than SLSQP's local search. Returns the best
   proportion vector and an estimate of uncertainty using the IntervalModel
   when available.

2. **Multi-objective Pareto** (Optuna NSGA-II) — finds the trade-off front
   between competing goals (e.g., maximise quality, minimise cost,
   minimise distance from a target alcohol level). Returns the full Pareto
   set so the enologist can pick a point that matches their priorities.

Proportions are sampled as raw weights in [0, 1] and then normalised to sum
to 1. This loses a degree of freedom but lets Optuna's standard samplers
operate without equality-constraint support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import optuna
import pandas as pd

from ..data.loader import FEATURE_COLUMNS
from .blend_recommender import BaseWine, BlendConstraints, mix_chemistry
from .quality_predictor import QualityModel
from .uncertainty import IntervalModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class BOBlendResult:
    proportions: dict
    blended_chemistry: dict
    predicted_quality: float
    quality_lo: float | None
    quality_hi: float | None
    cost: float | None
    alcohol: float


def _sample_proportions(trial: optuna.Trial, n: int) -> np.ndarray:
    raw = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n)])
    total = raw.sum()
    if total < 1e-6:
        return np.full(n, 1.0 / n)
    return raw / total


def _constraint_penalty(chem: dict, c: BlendConstraints) -> float:
    pen = 0.0
    if c.min_alcohol is not None:
        pen += max(c.min_alcohol - chem["alcohol"], 0.0) ** 2 * 50
    if c.max_alcohol is not None:
        pen += max(chem["alcohol"] - c.max_alcohol, 0.0) ** 2 * 50
    if c.max_volatile_acidity is not None:
        pen += max(chem["volatile acidity"] - c.max_volatile_acidity, 0.0) ** 2 * 100
    if c.max_total_so2 is not None:
        pen += max(chem["total sulfur dioxide"] - c.max_total_so2, 0.0) ** 2 * 0.05
    return pen


def recommend_bo(
    base_wines: Sequence[BaseWine],
    quality_model: QualityModel,
    constraints: BlendConstraints | None = None,
    interval_model: IntervalModel | None = None,
    n_trials: int = 80,
    seed: int = 42,
) -> BOBlendResult:
    """Single-objective BO using TPE.

    The constraint set is softened into the objective as a quadratic penalty
    so Optuna's sampler can still navigate when constraints are tight.
    """
    constraints = constraints or BlendConstraints()
    n = len(base_wines)

    def objective(trial: optuna.Trial) -> float:
        p = _sample_proportions(trial, n)
        chem = mix_chemistry(base_wines, p)
        q = float(quality_model.predict(chem)[0])
        pen = _constraint_penalty(chem, constraints)
        return -q + pen  # minimise

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    p = np.array([best_trial.params[f"w_{i}"] for i in range(n)])
    p = p / max(p.sum(), 1e-9)
    chem = mix_chemistry(base_wines, p)
    q = float(quality_model.predict(chem)[0])

    lo = hi = None
    if interval_model is not None:
        iv = interval_model.predict(chem)
        lo = float(iv["conformal_lo"][0])
        hi = float(iv["conformal_hi"][0])

    return BOBlendResult(
        proportions={w.name: float(pi) for w, pi in zip(base_wines, p)},
        blended_chemistry=chem,
        predicted_quality=q,
        quality_lo=lo,
        quality_hi=hi,
        cost=None,
        alcohol=chem["alcohol"],
    )


def pareto_front(
    base_wines: Sequence[BaseWine],
    quality_model: QualityModel,
    costs_per_base: Sequence[float] | None = None,
    target_alcohol: float | None = None,
    n_trials: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Multi-objective Pareto using NSGA-II.

    Objectives:
      1. Maximise predicted quality (minimised as -q)
      2. (if costs given) Minimise weighted cost of the blend
      3. (if target_alcohol given) Minimise |alcohol − target_alcohol|

    Returns a DataFrame of the Pareto-optimal points, one row per solution,
    with columns for proportions, blended chemistry summary and objectives.
    """
    n = len(base_wines)

    def objectives(trial: optuna.Trial) -> tuple[float, ...]:
        p = _sample_proportions(trial, n)
        chem = mix_chemistry(base_wines, p)
        q = float(quality_model.predict(chem)[0])
        objs: list[float] = [-q]
        if costs_per_base is not None:
            cost = float(np.dot(p, np.asarray(costs_per_base, dtype=float)))
            objs.append(cost)
        if target_alcohol is not None:
            objs.append(abs(chem["alcohol"] - target_alcohol))
        return tuple(objs)

    directions = ["minimize"] * (1 + int(costs_per_base is not None) + int(target_alcohol is not None))
    study = optuna.create_study(
        directions=directions, sampler=optuna.samplers.NSGAIISampler(seed=seed)
    )
    study.optimize(objectives, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for t in study.best_trials:
        p = np.array([t.params[f"w_{i}"] for i in range(n)])
        p = p / max(p.sum(), 1e-9)
        chem = mix_chemistry(base_wines, p)
        q = float(quality_model.predict(chem)[0])
        row: dict = {
            "qualidade": q,
            "álcool": chem["alcohol"],
            "acidez_volátil": chem["volatile acidity"],
            "SO2_total": chem["total sulfur dioxide"],
        }
        if costs_per_base is not None:
            row["custo"] = float(np.dot(p, np.asarray(costs_per_base)))
        if target_alcohol is not None:
            row["dist_alcool"] = abs(chem["alcohol"] - target_alcohol)
        for w, pi in zip(base_wines, p):
            row[f"%_{w.name}"] = float(round(pi * 100, 1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("qualidade", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from .blend_recommender import sample_base_wines_from_dataset
    from .quality_predictor import MODELS_DIR
    from ..data.loader import load_raw

    df = load_raw("both")
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    bases = sample_base_wines_from_dataset(df, "red", n=4)
    print("Single-objective BO:")
    r = recommend_bo(bases, model, n_trials=60)
    for k, v in r.proportions.items():
        print(f"  {k}: {v*100:.1f}%")
    print(f"  quality={r.predicted_quality:.3f}")

    print("\nMulti-objective Pareto (quality vs alcohol target=13):")
    pf = pareto_front(bases, model, target_alcohol=13.0, n_trials=120)
    print(pf.head(8).to_string(index=False))
