"""Optuna hyperparameter tuning.

Tunes RandomForest (the ablation winner) and XGBoost on UCI quality data
per wine type, and XGBoost on Wine Reviews for the market model. Saves best
params to models/best_params_*.json so train_all picks them up next time.

Each trial = full Stratified 5-Fold (UCI) or GroupKFold 5-Fold (market),
average R² as the objective.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBRegressor

from src.data.loader import load_raw
from src.data.preprocessor import add_engineered_features, basic_clean
from src.data.reviews_loader import load_reviews
from src.models.market_predictor import _build_pipeline, prepare_features
from src.models.quality_predictor import MODELS_DIR

N_TRIALS_QUALITY = 20
N_TRIALS_MARKET = 12
MARKET_SAMPLE = 15_000
N_SPLITS = 5
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xy_quality(wine_type: str):
    df = basic_clean(load_raw("both"))
    sub = df[df["type"] == wine_type].copy()
    sub = add_engineered_features(sub)
    feat_cols = [c for c in sub.columns if c not in {"quality", "type"}]
    return sub[feat_cols].values, sub["quality"].values


def _cv_r2_quality(estimator, X, y):
    bins = pd.cut(y, bins=[-0.5, 4.5, 5.5, 6.5, 7.5, 10.5], labels=False).astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, bins):
        est = estimator()
        est.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], est.predict(X[te])))
    return float(np.mean(scores))


def tune_quality_rf(wine_type: str) -> dict:
    X, y = _xy_quality(wine_type)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        return _cv_r2_quality(
            lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1, **params),
            X, y,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_QUALITY, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params,
            "n_trials": N_TRIALS_QUALITY, "model": "random_forest"}


def tune_quality_xgb(wine_type: str) -> dict:
    X, y = _xy_quality(wine_type)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }
        return _cv_r2_quality(
            lambda: XGBRegressor(
                random_state=SEED, n_jobs=-1, tree_method="hist",
                objective="reg:squarederror", **params,
            ),
            X, y,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_QUALITY, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params,
            "n_trials": N_TRIALS_QUALITY, "model": "xgboost"}


def tune_market() -> dict:
    df = load_reviews().sample(n=MARKET_SAMPLE, random_state=SEED).reset_index(drop=True)
    feats = prepare_features(df)
    y = df["points"].astype(float).values
    groups = df["winery"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(feats, y, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        }
        scores = []
        for tr, te in splits:
            pipe = _build_pipeline()
            # Inject params into the estimator
            pipe.named_steps["est"].set_params(**params)
            pipe.fit(feats.iloc[tr], y[tr])
            pred = pipe.predict(feats.iloc[te])
            scores.append(r2_score(y[te], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_MARKET, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params,
            "n_trials": N_TRIALS_MARKET, "model": "xgboost+tfidf",
            "n_samples": MARKET_SAMPLE}


def main() -> None:
    out: dict = {}
    for wt in ("red", "white"):
        print(f"[quality:{wt}] tuning RF ({N_TRIALS_QUALITY} trials)...")
        rf = tune_quality_rf(wt)
        print(f"  RF best R²={rf['best_value']:.4f}")
        print(f"[quality:{wt}] tuning XGB ({N_TRIALS_QUALITY} trials)...")
        xgb = tune_quality_xgb(wt)
        print(f"  XGB best R²={xgb['best_value']:.4f}")
        out[wt] = {"random_forest": rf, "xgboost": xgb,
                   "winner": "random_forest" if rf["best_value"] >= xgb["best_value"]
                   else "xgboost"}
        (MODELS_DIR / f"best_params_quality_{wt}.json").write_text(
            json.dumps(out[wt], indent=2), encoding="utf-8"
        )

    print(f"[market] tuning XGB+TF-IDF ({N_TRIALS_MARKET} trials)...")
    market = tune_market()
    print(f"  market best R²={market['best_value']:.4f}")
    out["market"] = market
    (MODELS_DIR / "best_params_market.json").write_text(
        json.dumps(market, indent=2), encoding="utf-8"
    )

    (MODELS_DIR / "best_params_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("OK. Best params saved under models/best_params_*.json")


if __name__ == "__main__":
    main()
