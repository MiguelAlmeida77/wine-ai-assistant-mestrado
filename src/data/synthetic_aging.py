"""Synthetic aging trajectories.

UCI Wine Quality has no temporal data, so we generate plausible aging
trajectories using deterministic equations grounded in enology literature.
These are first-order approximations meant to demonstrate the *shape* of
chemical and sensory evolution — they are intentionally simple so the thesis
can document and defend every assumption.

Key references underlying the equations:
 - Free SO₂ decay: pseudo-first-order, k ≈ 0.18 yr⁻¹ in well-corked bottles
   (Ribéreau-Gayon et al., "Handbook of Enology", vol. 2).
 - Volatile acidity rise: linear ~0.02 g/L per year for sound wines
   (Boulton et al., "Principles and Practices of Winemaking").
 - Total acidity drop via tartrate precipitation: exponential approach to
   asymptote, time constant ~3 years.
 - pH drift: small positive linear drift balancing acidity loss.
 - Anthocyanin decay (red): k ≈ 0.35 yr⁻¹, with proxy via fixed acidity.

Quality is modelled as an inverted-U curve: it rises while integration and
polymerization improve mouthfeel and complexity, peaks, and declines as
oxidation and SO₂ depletion take over. The peak year depends on initial
alcohol, acidity and (for reds) phenolic load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import FEATURE_COLUMNS, load_raw


@dataclass
class AgingParams:
    so2_decay_per_year: float = 0.18
    va_increase_per_year: float = 0.020
    ta_drop_fraction: float = 0.18      # asymptotic total drop
    ta_time_constant: float = 3.0
    ph_drift_per_year: float = 0.015
    residual_sugar_drop_per_year: float = 0.02
    chlorides_stable: bool = True
    sulphates_drop_per_year: float = 0.01


PARAMS = AgingParams()


def project_chemistry(initial: dict, years: float, wine_type: str) -> dict:
    """Project the chemical state of a wine `years` after bottling."""
    chem = dict(initial)
    t = max(years, 0.0)

    # Free SO2 exponential decay
    chem["free sulfur dioxide"] = initial["free sulfur dioxide"] * np.exp(
        -PARAMS.so2_decay_per_year * t
    )
    # Total SO2 follows free SO2 with floor (bound fraction roughly stable)
    bound = max(initial["total sulfur dioxide"] - initial["free sulfur dioxide"], 0.0)
    chem["total sulfur dioxide"] = chem["free sulfur dioxide"] + bound * np.exp(
        -0.05 * t
    )

    # Volatile acidity slow linear increase, capped
    chem["volatile acidity"] = min(
        initial["volatile acidity"] + PARAMS.va_increase_per_year * t,
        initial["volatile acidity"] + 0.5,
    )

    # Total acidity drop modelled as fixed acidity drop towards asymptote
    drop_factor = 1 - PARAMS.ta_drop_fraction * (1 - np.exp(-t / PARAMS.ta_time_constant))
    chem["fixed acidity"] = initial["fixed acidity"] * drop_factor

    # pH drifts up slightly (acid loss > buffer change)
    chem["pH"] = initial["pH"] + PARAMS.ph_drift_per_year * t

    # Sulphates decline slowly
    chem["sulphates"] = max(initial["sulphates"] - PARAMS.sulphates_drop_per_year * t, 0.1)

    # Residual sugar very slow drop (negligible in dry wines)
    chem["residual sugar"] = max(
        initial["residual sugar"] - PARAMS.residual_sugar_drop_per_year * t, 0.5
    )

    # Citric acid slowly metabolised by microflora
    chem["citric acid"] = max(initial["citric acid"] - 0.02 * t, 0.0)

    # Alcohol, chlorides, density approximately stable
    chem["alcohol"] = initial["alcohol"]
    chem["chlorides"] = initial["chlorides"]
    chem["density"] = initial["density"]

    if wine_type == "red":
        # Reds gain slight density from polymeric pigment accumulation (negligible);
        # we leave density flat for simplicity.
        pass

    return chem


def peak_year(initial: dict, wine_type: str) -> float:
    """Estimate the year at which quality peaks.

    Heuristic from enology lore:
      - Higher alcohol + fixed acidity + (for reds) phenolic body push the
        peak later.
      - Higher volatile acidity and lower SO₂ pull it earlier.
    """
    alcohol = initial["alcohol"]
    fa = initial["fixed acidity"]
    so2 = initial["free sulfur dioxide"]
    va = initial["volatile acidity"]

    base = 2.0 if wine_type == "white" else 4.0
    body = 0.6 * (alcohol - 10.0) + 0.4 * (fa - 7.0)
    preservation = 0.05 * (so2 - 20.0) - 4.0 * (va - 0.4)
    peak = base + body + preservation
    return float(np.clip(peak, 1.0, 20.0))


def quality_curve(
    initial: dict,
    initial_quality: float,
    wine_type: str,
    years_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Inverted-U quality curve sampled on a year grid.

    The curve approaches `initial_quality + gain` at the peak and decays
    afterwards with a tail that depends on phenolic content.
    """
    if years_grid is None:
        years_grid = np.arange(0.0, 15.5, 0.5)
    tp = peak_year(initial, wine_type)
    gain = 0.4 if wine_type == "white" else 0.8
    # Width controls how broad the plateau is; richer wines hold longer
    width = 2.5 if wine_type == "white" else 4.0
    decline_rate = 0.08 if wine_type == "white" else 0.05

    rows = []
    for t in years_grid:
        bump = gain * np.exp(-((t - tp) ** 2) / (2 * width**2))
        decline = decline_rate * max(t - tp, 0.0)
        chem = project_chemistry(initial, t, wine_type)
        # Penalise excessive volatile acidity past sensory threshold (~0.9 g/L)
        va_penalty = max(chem["volatile acidity"] - 0.9, 0.0) * 1.5
        quality_t = initial_quality + bump - decline - va_penalty
        row = {"year": float(t), "predicted_quality": float(quality_t), **chem}
        rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_dataset(
    n_per_type: int = 400, seed: int = 11
) -> pd.DataFrame:
    """Generate (chemistry × year × type) → quality training table.

    Used to fit a single regression model that learns to interpolate the
    synthetic equations and any composition effects of the underlying UCI
    samples (so the user can query arbitrary intermediate ages).
    """
    df = load_raw("both")
    rng = np.random.default_rng(seed)
    rows = []
    for wt in ("red", "white"):
        sub = df[df["type"] == wt].reset_index(drop=True)
        sample_idx = rng.choice(len(sub), size=min(n_per_type, len(sub)), replace=False)
        for k in sample_idx:
            row = sub.iloc[k]
            initial = {c: float(row[c]) for c in FEATURE_COLUMNS}
            curve = quality_curve(initial, float(row["quality"]), wt)
            curve["type"] = wt
            curve["sample_id"] = int(k)
            rows.append(curve)
    out = pd.concat(rows, ignore_index=True)
    return out


if __name__ == "__main__":
    ds = build_synthetic_dataset()
    out = load_raw("red").iloc[:0]  # noop, just keep imports tidy
    target = (load_raw("red").iloc[:0])  # noop
    print(f"Synthetic dataset rows: {len(ds)}")
    print(ds.head())
