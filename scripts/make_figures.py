"""Gera as figuras do Capítulo 5 como PNG de alta resolução.

Saída em `figures/`:
  fig_5_1_cobertura.png   — cobertura empírica vs alvo (incerteza)
  fig_5_2_pareto.png      — fronteira de Pareto (qualidade vs álcool)

Executar (a partir da raiz do projeto):
    .\\.venv\\Scripts\\python.exe scripts\\make_figures.py
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

from src.data.loader import load_raw
from src.models.blend_bo import pareto_front
from src.models.blend_recommender import sample_base_wines_from_dataset
from src.models.quality_predictor import MODELS_DIR, QualityModel

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Paleta enológica coerente com a app
WINE_RED = "#7a1f3a"
WINE_GREY = "#8a8a8a"
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_cobertura() -> Path:
    metrics = json.loads((MODELS_DIR / "interval_metrics.json").read_text(encoding="utf-8"))
    tipos = ["Tinto", "Branco"]
    quant = [metrics["red"]["quantile_xgb"]["empirical_coverage"] * 100,
             metrics["white"]["quantile_xgb"]["empirical_coverage"] * 100]
    conf = [metrics["red"]["conformal"]["empirical_coverage"] * 100,
            metrics["white"]["conformal"]["empirical_coverage"] * 100]

    x = np.arange(len(tipos))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - w / 2, quant, w, label="Regressão quantílica", color=WINE_GREY)
    b2 = ax.bar(x + w / 2, conf, w, label="Predição conformal", color=WINE_RED)
    ax.axhline(80, color="black", linestyle="--", linewidth=1)
    ax.text(-0.42, 81.5, "Alvo (80%)", fontsize=9, ha="left")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9)
    ax.set_xticks(x, tipos)
    ax.set_ylabel("Cobertura empírica (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cobertura empírica dos intervalos de predição (alvo 80%)")
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    out = FIG_DIR / "fig_5_1_cobertura.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_pareto() -> Path:
    """Fronteira de Pareto qualidade vs. teor alcoólico-alvo.

    Objetivos conflituantes: maximizar a qualidade prevista e minimizar a
    distância a um teor alcoólico-alvo. Escolhem-se vinhos-base com ampla
    amplitude de álcool e um alvo situado no extremo inferior da gama
    alcançável, de modo a gerar uma verdadeira tensão entre os dois objetivos
    (e, portanto, uma fronteira com dispersão real de soluções).
    """
    df = load_raw("both")
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    bases = sample_base_wines_from_dataset(df, "red", n=6, seed=3)

    # Amplitude de álcool dos vinhos-base disponíveis
    alcohols = [b.chemistry["alcohol"] for b in bases]
    alc_min, alc_max = min(alcohols), max(alcohols)
    # Alvo no extremo inferior: cria tensão contra a qualidade, que tende a
    # subir com o álcool → a fronteira espalha-se ao longo do eixo do álcool.
    target_alc = round(alc_min + 0.2, 1)

    pf = pareto_front(bases, model, target_alcohol=target_alc, n_trials=400)
    pf = pf.sort_values("álcool")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pf["álcool"], pf["qualidade"], color=WINE_GREY, linewidth=1,
            alpha=0.6, zorder=2)
    ax.scatter(pf["álcool"], pf["qualidade"], color=WINE_RED,
               s=55, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("Teor alcoólico do blend (% vol.)")
    ax.set_ylabel("Qualidade prevista")
    ax.set_title("Fronteira de Pareto: qualidade vs. teor alcoólico")
    fig.tight_layout()
    out = FIG_DIR / "fig_5_2_pareto.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_cobertura, fig_pareto):
        out = fn()
        print(f"  gerada: {out.relative_to(ROOT)}")
    print(f"\nTodas as figuras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
