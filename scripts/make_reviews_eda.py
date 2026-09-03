"""Figuras de análise exploratória do conjunto Wine Reviews.

Saída em `figures/`:
  fig_reviews_distribuicoes.png  — distribuição de pontuações e de preços
  fig_reviews_paises.png         — países mais representados
  fig_reviews_castas.png         — castas mais representadas
  fig_reviews_pontos_por_pais.png — pontuação por país (top produtores)

Executar (a partir da raiz do projeto):
    .\\.venv\\Scripts\\python.exe scripts\\make_reviews_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.reviews_loader import load_reviews

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

WINE_RED = "#7a1f3a"
WINE_GOLD = "#c9a227"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> Path:
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_distribuicoes(df) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Pontuações (80-100)
    axes[0].hist(df["points"], bins=range(80, 102), color=WINE_RED,
                 edgecolor="white", rwidth=0.9)
    axes[0].set_xlabel("Pontuação do crítico")
    axes[0].set_ylabel("Número de críticas")
    axes[0].set_title("Distribuição das pontuações")

    # Preço (escala log)
    price = df["price"].dropna()
    bins = np.logspace(np.log10(price.min()), np.log10(price.quantile(0.999)), 40)
    axes[1].hist(price, bins=bins, color=WINE_GOLD, edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Preço (USD, escala log)")
    axes[1].set_ylabel("Número de críticas")
    axes[1].set_title("Distribuição dos preços")
    axes[1].set_xticks([10, 30, 100, 300, 1000],
                       ["10", "30", "100", "300", "1000"])

    fig.suptitle("Wine Reviews — distribuição de pontuações e preços", y=1.03)
    return _save(fig, "fig_reviews_distribuicoes.png")


def fig_paises(df) -> Path:
    top = df["country"].value_counts().head(15).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(range(len(top)), top.values, color=WINE_RED)
    ax.bar_label(bars, labels=[f"{v:,}".replace(",", " ") for v in top.values],
                 padding=3, fontsize=8)
    ax.set_yticks(range(len(top)), top.index, fontsize=9)
    ax.set_xlabel("Número de críticas")
    ax.set_title("Países mais representados no Wine Reviews (top 15)")
    ax.margins(x=0.12)
    return _save(fig, "fig_reviews_paises.png")


def fig_castas(df) -> Path:
    top = df["variety"].value_counts().head(15).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(range(len(top)), top.values, color=WINE_GOLD)
    ax.bar_label(bars, labels=[f"{v:,}".replace(",", " ") for v in top.values],
                 padding=3, fontsize=8)
    ax.set_yticks(range(len(top)), top.index, fontsize=9)
    ax.set_xlabel("Número de críticas")
    ax.set_title("Castas / blends mais representados (top 15)")
    ax.margins(x=0.12)
    return _save(fig, "fig_reviews_castas.png")


def fig_pontos_por_pais(df) -> Path:
    top_countries = df["country"].value_counts().head(10).index.tolist()
    # Ordenar por mediana de pontuação para leitura mais clara
    order = (
        df[df["country"].isin(top_countries)]
        .groupby("country")["points"].median()
        .sort_values(ascending=True).index.tolist()
    )
    data = [df[df["country"] == c]["points"].values for c in order]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for box in bp["boxes"]:
        box.set(facecolor=WINE_RED, alpha=0.6)
    ax.set_yticks(range(1, len(order) + 1), order, fontsize=9)
    ax.set_xlabel("Pontuação do crítico (80–100)")
    ax.set_title("Distribuição das pontuações por país (10 mais representados)")
    return _save(fig, "fig_reviews_pontos_por_pais.png")


def main() -> None:
    print("A carregar Wine Reviews...")
    df = load_reviews()
    print(f"  {len(df):,} críticas")
    for fn in (fig_distribuicoes, fig_paises, fig_castas, fig_pontos_por_pais):
        out = fn(df)
        print(f"  gerada: {out.relative_to(ROOT)}")
    print(f"\nTodas as figuras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
