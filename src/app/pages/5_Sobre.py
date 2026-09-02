"""Página informativa: metodologia, métricas e limitações."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.models.quality_predictor import MODELS_DIR

INTERVAL_METRICS = MODELS_DIR / "interval_metrics.json"
MARKET_METRICS = MODELS_DIR / "market_metrics.json"
EVAL_REPORT = MODELS_DIR / "evaluation_report.json"
BEST_PARAMS = MODELS_DIR / "best_params_summary.json"

st.set_page_config(page_title="Sobre", layout="wide")
st.title("Sobre a Dissertação")

st.markdown(
    """
    ## Tema
    *Assistente Inteligente para Afinação e Blend de Vinhos.*

    Este protótipo demonstra a viabilidade de utilizar Inteligência Artificial
    para apoiar enólogos de adegas pequenas e médias em decisões técnicas
    críticas — predição de qualidade, otimização de blends, estimativa da
    aceitação de mercado e geração de notas de prova.

    ## Arquitetura
    """
)

st.code(
    """
src/
├── data/
│   ├── loader.py            # download e leitura UCI Wine Quality
│   ├── preprocessor.py      # limpeza + features derivadas
│   └── reviews_loader.py    # Wine Reviews (Wine Enthusiast)
├── models/
│   ├── quality_predictor.py # XGBoost / RandomForest por tipo de vinho
│   ├── uncertainty.py       # intervalos (quantile + conformal)
│   ├── blend_recommender.py # otimização SLSQP (scipy)
│   ├── blend_bo.py          # otimização bayesiana + Pareto (Optuna)
│   ├── market_predictor.py  # aceitação de mercado (XGBoost + TF-IDF)
│   └── tasting_notes.py     # geração de notas de prova
├── utils/
│   └── ranges.py            # ranges físicos para sliders
└── app/
    ├── app.py
    └── pages/               # 4 módulos da UI
""",
    language="text",
)

st.markdown(
    """
    ## Dados
    - **UCI Wine Quality** (Cortez et al., 2009) — 1 599 tintos + 4 898 brancos
      da região do Vinho Verde, 11 variáveis físico-químicas e nota sensorial
      média 0–10 de três provadores.
    - **Wine Reviews** (Wine Enthusiast, ~120 mil críticas) — país, casta,
      preço, provador e descritor textual, com pontuação 80–100 do crítico.
      Suporta os módulos de aceitação de mercado e notas de prova.

    ## Modelos e métricas
    """
)

# Load metrics if available
qmetrics_path = MODELS_DIR / "quality_metrics.json"

if qmetrics_path.exists():
    qmetrics = json.loads(qmetrics_path.read_text(encoding="utf-8"))
    rows = []
    for wt, m in qmetrics.items():
        win = m["winner"]
        rows.append(
            {
                "Tipo": "Tinto" if wt == "red" else "Branco",
                "Modelo vencedor": win,
                "R² (teste)": round(m[win]["r2"], 3),
                "RMSE": round(m[win]["rmse"], 3),
                "MAE": round(m[win]["mae"], 3),
                "N treino": m["n_train"],
                "N teste": m["n_test"],
            }
        )
    st.markdown("### Predição de qualidade")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown(
    """
    ### Quantificação de incerteza (intervalos 80%)
    """
)

if INTERVAL_METRICS.exists():
    iv = json.loads(INTERVAL_METRICS.read_text(encoding="utf-8"))
    rows = []
    for wt, m in iv.items():
        rows.append({
            "Tipo": "Tinto" if wt == "red" else "Branco",
            "Alvo": f"{m['target_coverage']:.0%}",
            "Cobertura — Quantile XGB": f"{m['quantile_xgb']['empirical_coverage']:.1%}",
            "Largura — Quantile XGB": round(m['quantile_xgb']['mean_width'], 2),
            "Cobertura — Conformal": f"{m['conformal']['empirical_coverage']:.1%}",
            "Largura — Conformal": round(m['conformal']['mean_width'], 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "*Conformal prediction* dá garantia teórica de cobertura sob trocabilidade; "
        "*Quantile XGB* é mais estreito mas tende a sub-cobrir em distribuições assimétricas."
    )

st.markdown("### Aceitação de mercado (Wine Reviews)")
if MARKET_METRICS.exists():
    mm = json.loads(MARKET_METRICS.read_text(encoding="utf-8"))
    st.dataframe(
        pd.DataFrame([{
            "R² (teste, GroupKFold winery)": round(mm["r2"], 3),
            "RMSE (escala 80–100)": round(mm["rmse"], 2),
            "MAE": round(mm["mae"], 2),
            "N treino": mm["n_train"],
            "N teste": mm["n_test"],
            "Wineries únicas em teste": mm["n_wineries_test"],
        }]),
        use_container_width=True, hide_index=True,
    )

st.markdown("### Validação cruzada e ablation")
if EVAL_REPORT.exists():
    rep = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    rows = []
    for wt, by_fs in rep["quality"].items():
        for fs, by_est in by_fs.items():
            for est, s in by_est.items():
                rows.append({
                    "Tipo": "Tinto" if wt == "red" else "Branco",
                    "Features": "Derivadas" if fs == "engineered" else "Brutas",
                    "Modelo": est,
                    "R² (μ ± σ)": f"{s['r2_mean']:.3f} ± {s['r2_std']:.3f}",
                    "RMSE": f"{s['rmse_mean']:.3f} ± {s['rmse_std']:.3f}",
                })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Stratified 5-Fold por bin de qualidade. Achados: (a) Random Forest "
        "iguala/supera XGBoost neste tamanho de dataset; (b) as features "
        "derivadas (ex.: rácio álcool/açúcar) não trazem ganho — as árvores já "
        "aprendem essas interações."
    )

st.markdown("### Hiperparâmetros (Optuna)")
if BEST_PARAMS.exists():
    bp = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))
    rows = []
    for wt in ("red", "white"):
        if wt in bp:
            block = bp[wt]
            rows.append({
                "Alvo": f"Qualidade — {('Tinto' if wt=='red' else 'Branco')}",
                "Vencedor": block["winner"],
                "R² CV": round(block[block["winner"]]["best_value"], 4),
                "Trials": block[block["winner"]]["n_trials"],
            })
    if "market" in bp:
        rows.append({
            "Alvo": "Mercado",
            "Vencedor": bp["market"]["model"],
            "R² CV": round(bp["market"]["best_value"], 4),
            "Trials": bp["market"]["n_trials"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Corra `scripts\\tune.py` para preencher esta secção.")

st.markdown(
    """
    ## Limitações conhecidas
    - O dataset UCI cobre apenas vinhos da região do *Vinho Verde* — a
      generalização para outras regiões/castas exige re-treino.
    - A qualidade UCI é a média de três provadores numa escala 0–10; é uma
      proxy razoável mas não substitui um painel sensorial dedicado.
    - As pontuações de crítica (Wine Reviews) provêm de uma única publicação e
      a escala é censurada à esquerda (não são publicados vinhos abaixo de 80),
      introduzindo enviesamento de seleção.
    - O modelo de blend assume mistura linear por fração volumétrica — exato
      para álcool, açúcar e SO₂; aproximado para pH e densidade.

    ## Próximos passos
    1. Integrar dados de provas sensoriais reais de uma adega parceira.
    2. Combinar química, texto e cor num modelo multimodal.
    3. Auditar o desempenho por região, casta e escalão de preço.
    """
)
