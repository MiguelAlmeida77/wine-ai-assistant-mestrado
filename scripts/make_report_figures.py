"""Gera figuras de exploração de dados e de resultados para a dissertação.

Saída em `figures/`. Todas as figuras são produzidas a partir dos dados e
modelos reais do projeto.

Executar (a partir da raiz do projeto):
    .\\.venv\\Scripts\\python.exe scripts\\make_report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data.loader import FEATURE_COLUMNS, load_raw
from src.data.preprocessor import add_engineered_features, basic_clean
from src.models.quality_predictor import MODELS_DIR, QualityModel

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

WINE_RED = "#7a1f3a"
WINE_GOLD = "#c9a227"
WINE_GREY = "#8a8a8a"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Rótulos curtos para gráficos densos (heatmaps)
SHORT_LABELS = {
    "fixed acidity": "Acidez fixa",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ác. cítrico",
    "residual sugar": "Açúcar res.",
    "chlorides": "Cloretos",
    "free sulfur dioxide": "SO₂ livre",
    "total sulfur dioxide": "SO₂ total",
    "density": "Densidade",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Álcool",
    "quality": "Qualidade",
    # Features derivadas (engenharia de atributos)
    "total_acidity": "Acidez total",
    "bound_sulfur_dioxide": "SO₂ combinado",
    "alcohol_sugar_ratio": "Rácio álcool/açúcar",
}


def _save(fig, name: str) -> Path:
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- EDA / dados
def fig_dist_qualidade() -> Path:
    df = load_raw("both")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.arange(2.5, 9.5, 1)
    red = df[df["type"] == "red"]["quality"]
    white = df[df["type"] == "white"]["quality"]
    ax.hist([red, white], bins=bins, label=["Tinto", "Branco"],
            color=[WINE_RED, WINE_GOLD], edgecolor="white", rwidth=0.9)
    ax.set_xlabel("Nota de qualidade sensorial")
    ax.set_ylabel("Número de amostras")
    ax.set_title("Distribuição das notas de qualidade (UCI Wine Quality)")
    ax.legend(frameon=False)
    return _save(fig, "fig_dist_qualidade.png")


def _corr_heatmap(wine_type: str, color: str, fname: str) -> Path:
    df = load_raw(wine_type)
    cols = FEATURE_COLUMNS + ["quality"]
    corr = df[cols].corr()
    labels = [SHORT_LABELS[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    # Anotar valores
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficiente de correlação de Pearson")
    tipo = "tinto" if wine_type == "red" else "branco"
    ax.set_title(f"Matriz de correlação — vinho {tipo}")
    return _save(fig, fname)


def fig_correlacao_tinto() -> Path:
    return _corr_heatmap("red", WINE_RED, "fig_correlacao_tinto.png")


def fig_correlacao_branco() -> Path:
    return _corr_heatmap("white", WINE_GOLD, "fig_correlacao_branco.png")


def fig_alcool_qualidade() -> Path:
    df = load_raw("both")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (wt, color, tipo) in zip(
        axes, [("red", WINE_RED, "Tinto"), ("white", WINE_GOLD, "Branco")]
    ):
        sub = df[df["type"] == wt]
        qualities = sorted(sub["quality"].unique())
        data = [sub[sub["quality"] == q]["alcohol"].values for q in qualities]
        bp = ax.boxplot(data, positions=qualities, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.6)
        ax.set_xlabel("Nota de qualidade")
        ax.set_title(f"{tipo}")
    axes[0].set_ylabel("Teor alcoólico (% vol.)")
    fig.suptitle("Relação entre teor alcoólico e qualidade sensorial", y=1.02)
    return _save(fig, "fig_alcool_qualidade.png")


# ------------------------------------------------------------- resultados
def fig_importancia_qualidade() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (wt, color, tipo) in zip(
        axes, [("red", WINE_RED, "Tinto"), ("white", WINE_GOLD, "Branco")]
    ):
        model = QualityModel.load(MODELS_DIR / f"quality_{wt}.joblib")
        imp = getattr(model.estimator, "feature_importances_", None)
        if imp is None:
            continue
        order = np.argsort(imp)
        feats = [model.feature_cols[i] for i in order]
        labels = [SHORT_LABELS.get(f, f) for f in feats]
        ax.barh(range(len(feats)), imp[order], color=color)
        ax.set_yticks(range(len(feats)), labels, fontsize=8)
        ax.set_xlabel("Importância relativa")
        ax.set_title(f"{tipo} ({model.metrics['winner']})")
    fig.suptitle("Importância das variáveis na predição de qualidade", y=1.02)
    return _save(fig, "fig_importancia_qualidade.png")


def fig_pred_vs_real() -> Path:
    df = load_raw("both")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (wt, color, tipo) in zip(
        axes, [("red", WINE_RED, "Tinto"), ("white", WINE_GOLD, "Branco")]
    ):
        model = QualityModel.load(MODELS_DIR / f"quality_{wt}.joblib")
        sub = basic_clean(df[df["type"] == wt])
        sub = add_engineered_features(sub)
        X = sub[model.feature_cols]
        y = sub["quality"].values
        est = clone(model.estimator)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bins = pd.cut(y, bins=[-0.5, 4.5, 5.5, 6.5, 7.5, 10.5], labels=False)
        pred = cross_val_predict(est, X, y, cv=cv.split(X, bins), n_jobs=-1)
        # jitter para revelar densidade (as notas reais são inteiras)
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(y))
        ax.scatter(y + jitter, pred, s=8, alpha=0.25, color=color)
        lims = [2.5, 9]
        ax.plot(lims, lims, "--", color="black", linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Qualidade real")
        ax.set_title(f"{tipo}")
    axes[0].set_ylabel("Qualidade prevista (validação cruzada)")
    fig.suptitle("Qualidade prevista vs. real (predições fora-de-fold)", y=1.02)
    return _save(fig, "fig_pred_vs_real.png")


def fig_ablation() -> Path:
    rep = json.loads((MODELS_DIR / "evaluation_report.json").read_text(encoding="utf-8"))
    ests = ["baseline_mean", "random_forest", "gradient_boosting", "xgboost"]
    est_labels = ["Referência", "Random Forest", "Gradient Boosting", "XGBoost"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ests))
    w = 0.38
    red_r2 = [rep["quality"]["red"]["raw"][e]["r2_mean"] for e in ests]
    white_r2 = [rep["quality"]["white"]["raw"][e]["r2_mean"] for e in ests]
    red_err = [rep["quality"]["red"]["raw"][e]["r2_std"] for e in ests]
    white_err = [rep["quality"]["white"]["raw"][e]["r2_std"] for e in ests]
    ax.bar(x - w / 2, red_r2, w, yerr=red_err, capsize=3, label="Tinto", color=WINE_RED)
    ax.bar(x + w / 2, white_r2, w, yerr=white_err, capsize=3, label="Branco", color=WINE_GOLD)
    ax.set_xticks(x, est_labels, rotation=15)
    ax.set_ylabel("R² (validação cruzada 5-fold)")
    ax.set_title("Comparação de algoritmos na predição de qualidade")
    ax.legend(frameon=False)
    return _save(fig, "fig_ablation.png")


def fig_augmentation() -> Path:
    comp = json.loads((MODELS_DIR / "augmentation_comparison.json").read_text(encoding="utf-8"))
    conds = ["baseline_stratified", "augmented_grouped", "augmented_stratified_leaky"]
    cond_labels = ["Sem augmentação", "Com augmentação\n(GroupKFold)", "Com augmentação\n(aleatória, c/ fuga)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(conds))
    w = 0.38
    red_r2 = [comp["red"][c]["r2_mean"] for c in conds]
    white_r2 = [comp["white"][c]["r2_mean"] for c in conds]
    red_err = [comp["red"][c]["r2_std"] for c in conds]
    white_err = [comp["white"][c]["r2_std"] for c in conds]
    ax.bar(x - w / 2, red_r2, w, yerr=red_err, capsize=3, label="Tinto", color=WINE_RED)
    ax.bar(x + w / 2, white_r2, w, yerr=white_err, capsize=3, label="Branco", color=WINE_GOLD)
    ax.set_xticks(x, cond_labels, fontsize=9)
    ax.set_ylabel("R² (validação cruzada)")
    ax.set_title("Augmentação varietal: o efeito da fuga de informação")
    ax.legend(frameon=False)
    return _save(fig, "fig_augmentation.png")


def fig_descritores() -> Path:
    meta = json.loads((MODELS_DIR / "notes_metrics.json").read_text(encoding="utf-8"))
    freq = meta["frequencies"]
    # Mostrar TODOS os descritores retidos (frequência >= 1%), ordenados
    s = pd.Series(freq).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 9))
    bars = ax.barh(range(len(s)), s.values * 100, color=WINE_RED)
    ax.bar_label(bars, labels=[f"{v*100:.1f}" for v in s.values],
                 padding=2, fontsize=7)
    ax.set_yticks(range(len(s)), s.index, fontsize=8)
    ax.set_xlabel("Frequência nas críticas (%)")
    ax.set_title(f"Frequência dos {len(s)} descritores extraídos (Wine Reviews)")
    ax.margins(x=0.10)
    return _save(fig, "fig_descritores.png")


def fig_mercado_preco_pontos() -> Path:
    path = ROOT / "data" / "processed" / "reviews_sample.parquet"
    df = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.hexbin(np.log10(df["price"]), df["points"], gridsize=30, cmap="magma_r",
                   mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Nº de vinhos")
    ax.set_xlabel("Preço (USD, escala log)")
    ax.set_ylabel("Pontuação do crítico (80–100)")
    ax.set_title("Relação preço–pontuação (Wine Reviews)")
    # Converter ticks log para valores reais
    ticks = [1, 1.5, 2, 2.5, 3]
    ax.set_xticks(ticks, [f"{10**t:.0f}" for t in ticks])
    return _save(fig, "fig_mercado_preco_pontos.png")


def main() -> None:
    figs = [
        fig_dist_qualidade,
        fig_correlacao_tinto,
        fig_correlacao_branco,
        fig_alcool_qualidade,
        fig_importancia_qualidade,
        fig_pred_vs_real,
        fig_ablation,
        fig_augmentation,
        fig_descritores,
        fig_mercado_preco_pontos,
    ]
    for fn in figs:
        out = fn()
        print(f"  gerada: {out.relative_to(ROOT)}")
    print(f"\nTodas as figuras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
