"""Cross-validated evaluation + ablation studies.

Produces a JSON report (and a Markdown table) summarising:

* Stratified K-Fold by quality bin for the UCI quality predictor.
* Group K-Fold by winery for the market predictor (winery never seen in train
  also appears in test → tests cross-winery generalisation).
* Ablation: with vs without engineered features, XGB vs RandomForest vs
  GradientBoosting vs mean baseline.

These results feed straight into the dissertation's results chapter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBRegressor

from ..data.loader import FEATURE_COLUMNS, load_raw
from ..data.preprocessor import add_engineered_features, basic_clean
from .market_predictor import (
    CAT_COLS,
    MarketModel,
    _build_pipeline,
    prepare_features,
)
from .quality_predictor import MODELS_DIR


@dataclass
class FoldResult:
    rmse: float
    mae: float
    r2: float


def _metrics(y_true, y_pred) -> FoldResult:
    return FoldResult(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def _summary(fold_results: list[FoldResult]) -> dict:
    arr_rmse = np.array([r.rmse for r in fold_results])
    arr_mae = np.array([r.mae for r in fold_results])
    arr_r2 = np.array([r.r2 for r in fold_results])
    return {
        "rmse_mean": float(arr_rmse.mean()),
        "rmse_std": float(arr_rmse.std()),
        "mae_mean": float(arr_mae.mean()),
        "mae_std": float(arr_mae.std()),
        "r2_mean": float(arr_r2.mean()),
        "r2_std": float(arr_r2.std()),
        "n_folds": int(len(fold_results)),
    }


def _make_estimator(name: str, seed: int = 42):
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, tree_method="hist", objective="reg:squarederror",
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=seed, n_jobs=-1
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=seed,
        )
    if name == "baseline_mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown estimator: {name}")


def cv_quality_predictor(
    n_splits: int = 5, seed: int = 42
) -> dict:
    """Stratified K-Fold per wine type × estimator × feature set."""
    df = load_raw("both")
    results: dict = {}
    for wt in ("red", "white"):
        sub = df[df["type"] == wt].copy()
        sub = basic_clean(sub)
        results[wt] = {}
        for feature_set in ("raw", "engineered"):
            if feature_set == "engineered":
                data = add_engineered_features(sub)
            else:
                data = sub.copy()
            feat_cols = [c for c in data.columns if c not in {"quality", "type"}]
            X = data[feat_cols].values
            y = data["quality"].values
            bins = pd.cut(y, bins=[-0.5, 4.5, 5.5, 6.5, 7.5, 10.5],
                          labels=False).astype(int)
            results[wt][feature_set] = {}
            for est_name in ("baseline_mean", "random_forest",
                             "gradient_boosting", "xgboost"):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
                fold_results = []
                for tr, te in skf.split(X, bins):
                    est = _make_estimator(est_name, seed=seed)
                    est.fit(X[tr], y[tr])
                    pred = est.predict(X[te])
                    fold_results.append(_metrics(y[te], pred))
                results[wt][feature_set][est_name] = _summary(fold_results)
    return results


def cv_market_predictor(n_splits: int = 5, seed: int = 42,
                        sample: int | None = 40_000) -> dict:
    """Group K-Fold by winery on Wine Reviews."""
    from ..data.reviews_loader import load_reviews

    df = load_reviews()
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    feats = prepare_features(df)
    y = df["points"].astype(float).values
    groups = df["winery"].values

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    for tr, te in gkf.split(feats, y, groups=groups):
        pipe = _build_pipeline()
        pipe.fit(feats.iloc[tr], y[tr])
        pred = pipe.predict(feats.iloc[te])
        fold_results.append(_metrics(y[te], pred))
    return {
        "model": "xgboost+tfidf",
        "n_samples_used": int(len(df)),
        "summary": _summary(fold_results),
    }


def to_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Relatório de Avaliação\n")
    lines.append("## Preditor de qualidade (UCI, Stratified 5-Fold)\n")
    lines.append("| Tipo | Features | Modelo | R²  (μ±σ) | RMSE (μ±σ) | MAE (μ±σ) |")
    lines.append("|------|----------|--------|-----------|------------|-----------|")
    q = report["quality"]
    for wt in ("red", "white"):
        for fs in ("raw", "engineered"):
            for est in ("baseline_mean", "random_forest",
                        "gradient_boosting", "xgboost"):
                s = q[wt][fs][est]
                lines.append(
                    f"| {wt} | {fs} | {est} | "
                    f"{s['r2_mean']:.3f} ± {s['r2_std']:.3f} | "
                    f"{s['rmse_mean']:.3f} ± {s['rmse_std']:.3f} | "
                    f"{s['mae_mean']:.3f} ± {s['mae_std']:.3f} |"
                )

    m = report["market"]
    lines.append("\n## Preditor de mercado (Wine Reviews, GroupKFold por winery)\n")
    s = m["summary"]
    lines.append(f"Modelo: **{m['model']}** • amostras: {m['n_samples_used']:,}\n")
    lines.append("| Métrica | μ ± σ |")
    lines.append("|---------|-------|")
    lines.append(f"| R²      | {s['r2_mean']:.3f} ± {s['r2_std']:.3f} |")
    lines.append(f"| RMSE    | {s['rmse_mean']:.3f} ± {s['rmse_std']:.3f} |")
    lines.append(f"| MAE     | {s['mae_mean']:.3f} ± {s['mae_std']:.3f} |")
    return "\n".join(lines)


def run_all(out_json: Path | None = None,
            out_md: Path | None = None,
            n_splits: int = 5,
            market_sample: int | None = 40_000) -> dict:
    print("[1/2] CV quality predictor (UCI)...")
    quality = cv_quality_predictor(n_splits=n_splits)
    print("[2/2] CV market predictor (Wine Reviews)...")
    market = cv_market_predictor(n_splits=n_splits, sample=market_sample)
    report = {"quality": quality, "market": market}
    out_json = out_json or (MODELS_DIR / "evaluation_report.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report)
    out_md = out_md or (MODELS_DIR / "evaluation_report.md")
    out_md.write_text(md, encoding="utf-8")
    return report


if __name__ == "__main__":
    run_all()
    print("OK. Reports saved to models/evaluation_report.{json,md}")
