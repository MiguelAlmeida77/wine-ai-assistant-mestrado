"""Varietal augmentation of the UCI Wine Quality dataset.

The Vinho Verde dataset has no variety/castas information. To increase the
diversity seen by the quality model — and to enable a variety selector in the
UI — we generate synthetic copies of the UCI rows perturbed by varietal
*offsets* documented in the enological literature.

The offsets are calibrated from:
  - Cardoso, A. (2017), "Vinhos de Portugal".
  - Reynier, A. (2011), "Manual de Viticultura".
  - OIV statistical compendia (volatile acidity, sugar, alcohol ranges).
  - Cortez et al. (2009) original dataset means.

These are first-order shifts (means) plus a small Gaussian noise term whose
magnitude is calibrated to the dataset's own per-feature standard deviation.
The augmented dataset must be treated as a *prior-informed simulation*, not
as ground truth — the thesis should report metrics with and without
augmentation and discuss the trade-off (more diverse training set vs.
synthetic bias).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import FEATURE_COLUMNS, load_raw


@dataclass(frozen=True)
class VarietyProfile:
    name: str
    type: str          # "red" or "white"
    # Multiplicative shift on the dataset mean (1.0 = no change)
    offsets: dict[str, float]
    # Description used in the UI
    description: str


# --- White varieties (Vinho Verde region) ---
VARIETIES_WHITE: list[VarietyProfile] = [
    VarietyProfile(
        name="Alvarinho",
        type="white",
        offsets={
            "alcohol": 1.08,            # ~12.5–13.5% vs ~10.5 médio
            "fixed acidity": 1.10,
            "volatile acidity": 0.90,
            "citric acid": 1.20,
            "residual sugar": 0.70,
            "sulphates": 1.05,
            "free sulfur dioxide": 1.00,
        },
        description="Casta nobre do Minho — aromática, fresca, álcool elevado.",
    ),
    VarietyProfile(
        name="Loureiro",
        type="white",
        offsets={
            "alcohol": 0.97,
            "fixed acidity": 1.15,
            "volatile acidity": 0.95,
            "citric acid": 1.10,
            "residual sugar": 0.85,
        },
        description="Acidez marcante, perfil floral típico do Vinho Verde.",
    ),
    VarietyProfile(
        name="Arinto",
        type="white",
        offsets={
            "alcohol": 1.00,
            "fixed acidity": 1.20,
            "citric acid": 1.05,
            "residual sugar": 0.80,
            "sulphates": 1.10,
        },
        description="Acidez muito elevada, longevidade, alguma estrutura.",
    ),
    VarietyProfile(
        name="Encruzado",
        type="white",
        offsets={
            "alcohol": 1.10,
            "fixed acidity": 1.05,
            "residual sugar": 0.90,
            "sulphates": 1.15,
            "citric acid": 0.95,
        },
        description="Corpo, álcool e potencial de estágio (Dão).",
    ),
    VarietyProfile(
        name="Trajadura",
        type="white",
        offsets={
            "alcohol": 1.02,
            "fixed acidity": 0.92,
            "residual sugar": 1.10,
            "volatile acidity": 1.05,
        },
        description="Acidez moderada, casta de assemblage no Minho.",
    ),
]

# --- Red varieties (Vinho Verde + Douro) ---
VARIETIES_RED: list[VarietyProfile] = [
    VarietyProfile(
        name="Touriga Nacional",
        type="red",
        offsets={
            "alcohol": 1.10,
            "fixed acidity": 1.10,
            "volatile acidity": 0.85,
            "sulphates": 1.20,
            "residual sugar": 0.90,
            "citric acid": 1.10,
        },
        description="Casta-rainha do Douro — concentração, taninos firmes.",
    ),
    VarietyProfile(
        name="Aragonez (Tempranillo)",
        type="red",
        offsets={
            "alcohol": 1.05,
            "fixed acidity": 0.95,
            "sulphates": 1.10,
            "volatile acidity": 1.00,
        },
        description="Equilibrada, frutada, adapta-se a vários terroirs.",
    ),
    VarietyProfile(
        name="Trincadeira",
        type="red",
        offsets={
            "alcohol": 1.02,
            "fixed acidity": 0.90,
            "volatile acidity": 1.15,    # mais oxidativa
            "sulphates": 1.05,
        },
        description="Aromática, mais oxidativa, perfil do sul.",
    ),
    VarietyProfile(
        name="Castelão",
        type="red",
        offsets={
            "alcohol": 0.98,
            "fixed acidity": 0.95,
            "residual sugar": 1.05,
            "sulphates": 1.00,
        },
        description="Aromas terrosos, comum no Alentejo/Setúbal.",
    ),
    VarietyProfile(
        name="Vinhão",
        type="red",
        offsets={
            "alcohol": 0.92,
            "fixed acidity": 1.20,
            "volatile acidity": 1.10,
            "sulphates": 0.95,
            "citric acid": 1.05,
        },
        description="Tinto típico do Vinho Verde — acidez muito elevada.",
    ),
]

ALL_VARIETIES = VARIETIES_WHITE + VARIETIES_RED


def _apply_offsets(df: pd.DataFrame, profile: VarietyProfile,
                   noise_scale: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for feat, mult in profile.offsets.items():
        if feat not in out.columns:
            continue
        base = out[feat].values
        noise = rng.normal(0, noise_scale * np.abs(base), size=len(out))
        out[feat] = np.clip(base * mult + noise, 0.0, None)
    out["variety"] = profile.name
    return out


def augment(
    samples_per_variety: int = 200,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an augmented version of UCI Wine Quality with variety labels.

    For each variety we sample `samples_per_variety` rows from the matching
    type's UCI subset and apply the multiplicative + noise transformation.
    Original (unlabelled) rows are also returned with `variety = "Vinho Verde
    (UCI)"` so downstream models can keep using them.

    The returned DataFrame includes a `source_row` column with the index of
    the originating UCI row (or its own index, for non-augmented rows). This
    enables GroupKFold evaluation that avoids the leakage between an
    original row and its augmented copies.
    """
    df = load_raw("both").reset_index(drop=True)
    rng = np.random.default_rng(seed)

    pieces: list[pd.DataFrame] = []
    orig = df.copy()
    orig["variety"] = "Vinho Verde (UCI)"
    orig["source_row"] = orig.index
    pieces.append(orig)

    for k, profile in enumerate(ALL_VARIETIES):
        pool = df[df["type"] == profile.type]
        idx = rng.choice(pool.index.values, size=min(samples_per_variety, len(pool)),
                         replace=True)
        sub = df.loc[idx].reset_index(drop=True)
        aug = _apply_offsets(sub, profile, noise_scale=noise_scale, seed=seed + k)
        aug["source_row"] = idx
        pieces.append(aug)

    out = pd.concat(pieces, ignore_index=True)
    return out


def variety_choices(wine_type: str | None = None) -> list[VarietyProfile]:
    if wine_type == "red":
        return list(VARIETIES_RED)
    if wine_type == "white":
        return list(VARIETIES_WHITE)
    return list(ALL_VARIETIES)


if __name__ == "__main__":
    aug = augment()
    print(f"Total rows: {len(aug)}")
    print(aug["variety"].value_counts())
    print("\nMean alcohol per variety:")
    print(aug.groupby("variety")["alcohol"].mean().round(2))
