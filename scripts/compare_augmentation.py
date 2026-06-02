"""Compara métricas dos modelos de qualidade com e sem augmentação varietal.

Resultado guardado em `models/augmentation_comparison.json` e impresso na
consola. Útil para a tese justificar (ou não) a adoção de dados aumentados.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.data.loader import FEATURE_COLUMNS, load_raw
from src.data.preprocessor import add_engineered_features, basic_clean
from src.data.varietal_augmentation import augment
from src.models.quality_predictor import MODELS_DIR

N_SPLITS = 5
SEED = 42


def _eval_cv_stratified(X: np.ndarray, y: np.ndarray, bins: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return _fit_eval(skf.split(X, bins), X, y)


def _eval_cv_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    gkf = GroupKFold(n_splits=N_SPLITS)
    return _fit_eval(gkf.split(X, y, groups=groups), X, y)


def _fit_eval(splits, X, y) -> dict:
    rmses, maes, r2s = [], [], []
    for tr, te in splits:
        m = RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
        r2s.append(r2_score(y[te], p))
    return {
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
    }


def _build_xy(df: pd.DataFrame, wine_type: str, return_groups: bool = False):
    sub = df[df["type"] == wine_type].copy()
    sub = basic_clean(sub)
    sub = add_engineered_features(sub)
    drop = {"quality", "type", "variety", "source_row"}
    feat_cols = [c for c in sub.columns if c not in drop]
    X = sub[feat_cols].values
    y = sub["quality"].values
    bins = pd.cut(y, bins=[-0.5, 4.5, 5.5, 6.5, 7.5, 10.5], labels=False).astype(int)
    if return_groups:
        groups = sub["source_row"].values if "source_row" in sub.columns else np.arange(len(sub))
        return X, y, bins, groups
    return X, y, bins


def main() -> None:
    base = load_raw("both")
    aug_full = augment()
    # We have to split: original UCI rows are also in aug_full with label "Vinho Verde (UCI)"
    aug_only = aug_full[aug_full["variety"] != "Vinho Verde (UCI)"].copy()

    # Re-derive merged with source_row tracking
    aug_full2 = augment()  # has source_row for every row
    aug_only_g = aug_full2[aug_full2["variety"] != "Vinho Verde (UCI)"].copy()

    results: dict = {}
    for wt in ("red", "white"):
        print(f"\n=== {wt.upper()} ===")
        X, y, bins = _build_xy(base, wt)
        base_metrics = _eval_cv_stratified(X, y, bins)
        print(f"  baseline (stratified)        R² = {base_metrics['r2_mean']:.3f} ± {base_metrics['r2_std']:.3f}")

        # Merged with source_row column for group-aware CV
        base_g = base.copy().reset_index(drop=True)
        base_g["source_row"] = base_g.index
        merged_g = pd.concat([base_g, aug_only_g], ignore_index=True)
        Xa, ya, ba, ga = _build_xy(merged_g, wt, return_groups=True)
        aug_grouped = _eval_cv_grouped(Xa, ya, ga)
        print(f"  augmented (GroupKFold)        R² = {aug_grouped['r2_mean']:.3f} ± {aug_grouped['r2_std']:.3f} "
              f"(n={len(Xa)} vs {len(X)})")

        # Naive comparison (stratified, with leakage between original and copies)
        aug_naive = _eval_cv_stratified(Xa, ya, ba)
        print(f"  augmented (stratified, leaky) R² = {aug_naive['r2_mean']:.3f} ± {aug_naive['r2_std']:.3f}")

        results[wt] = {
            "n_baseline": int(len(X)),
            "n_augmented": int(len(Xa)),
            "baseline_stratified": base_metrics,
            "augmented_grouped": aug_grouped,
            "augmented_stratified_leaky": aug_naive,
            "delta_r2_clean": aug_grouped["r2_mean"] - base_metrics["r2_mean"],
            "delta_r2_leaky": aug_naive["r2_mean"] - base_metrics["r2_mean"],
        }

    out = MODELS_DIR / "augmentation_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
