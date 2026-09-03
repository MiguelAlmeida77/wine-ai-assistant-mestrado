"""Gera o diagrama de arquitetura em três camadas para a dissertação.

Saída: figures/fig_arquitetura.png

Executar (a partir da raiz do projeto):
    .\\.venv\\Scripts\\python.exe scripts\\make_architecture.py
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

WINE_RED = "#7a1f3a"
WINE_GOLD = "#c9a227"
SLATE = "#34495e"
BAND_DATA = "#f6efe3"
BAND_MODELS = "#f3e3e9"
BAND_PRESENT = "#e9edf1"

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 10})


def box(ax, x, y, w, h, text, facecolor, textcolor="white", fontsize=9.5, bold=False):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0, facecolor=facecolor,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize,
            fontweight="bold" if bold else "normal")


def band(ax, y, h, color, title):
    ax.add_patch(FancyBboxPatch(
        (0.02, y), 0.96, h,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=0, facecolor=color,
    ))
    ax.text(0.045, y + h - 0.022, title, ha="left", va="top",
            fontsize=12, fontweight="bold", color="#333333")


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color="#666666",
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- Camada de Dados ----------------
    band(ax, 0.80, 0.17, BAND_DATA, "CAMADA DE DADOS")
    box(ax, 0.07, 0.815, 0.26, 0.085, "UCI Wine Quality\n(6 497 amostras)", WINE_GOLD, "black")
    box(ax, 0.37, 0.815, 0.26, 0.085, "Wine Reviews\n(~120 mil críticas)", WINE_GOLD, "black")
    box(ax, 0.67, 0.815, 0.26, 0.085, "Pré-processamento\ne engenharia de atributos", WINE_GOLD, "black")

    # ---------------- Camada de Modelos ----------------
    band(ax, 0.31, 0.44, BAND_MODELS, "CAMADA DE MODELOS")
    # Linha 1
    box(ax, 0.06, 0.585, 0.27, 0.095, "Predição de qualidade\n(XGBoost / Random Forest)", WINE_RED)
    box(ax, 0.365, 0.585, 0.27, 0.095, "Quantificação de incerteza\n(quantile + conformal)", WINE_RED)
    box(ax, 0.67, 0.585, 0.27, 0.095, "Recomendação de blends\n(SLSQP + BO / Pareto)", WINE_RED)
    # Linha 2
    box(ax, 0.06, 0.455, 0.27, 0.095, "Explicabilidade\n(SHAP + contrafactuais)", WINE_RED)
    box(ax, 0.365, 0.455, 0.27, 0.095, "Aceitação de mercado\n(XGBoost + TF-IDF)", WINE_RED)
    box(ax, 0.67, 0.455, 0.27, 0.095, "Geração de notas de prova\n(multi-etiqueta)", WINE_RED)
    ax.text(0.5, 0.345, "Modelos treinados e persistidos em disco (joblib)",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    # ---------------- Camada de Apresentação ----------------
    band(ax, 0.04, 0.20, BAND_PRESENT, "CAMADA DE APRESENTAÇÃO")
    box(ax, 0.07, 0.135, 0.86, 0.055,
        "Interface interativa — Streamlit (aplicação web multipágina, em português)",
        SLATE, "white", fontsize=10.5, bold=True)
    pages = ["Predição de\nQualidade", "Recomendação\nde Blends",
             "Aceitação de\nMercado", "Notas de\nProva", "Sobre"]
    n = len(pages)
    pw, gap = 0.155, 0.02
    total = n * pw + (n - 1) * gap
    x0 = (1 - total) / 2
    for i, pg in enumerate(pages):
        box(ax, x0 + i * (pw + gap), 0.055, pw, 0.06, pg, "#5d6d7e", "white", fontsize=8.5)

    # ---------------- Setas entre camadas ----------------
    for x in (0.25, 0.5, 0.75):
        arrow(ax, x, 0.798, 0.688)   # dados -> modelos
        arrow(ax, x, 0.328, 0.245)   # modelos -> apresentação

    ax.set_title("Arquitetura do sistema em três camadas", fontsize=14,
                 fontweight="bold", pad=12)

    out = FIG_DIR / "fig_arquitetura.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"gerada: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
