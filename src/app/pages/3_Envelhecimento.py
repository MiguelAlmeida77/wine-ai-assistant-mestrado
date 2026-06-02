"""Módulo 3: Previsão da evolução do vinho durante o estágio em garrafa."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data.loader import FEATURE_COLUMNS, FEATURE_LABELS_PT
from src.data.synthetic_aging import peak_year
from src.models.aging_predictor import AgingModel
from src.models.quality_predictor import MODELS_DIR
from src.utils.ranges import default_chemistry, feature_ranges

st.set_page_config(page_title="Envelhecimento", layout="wide")
st.title("Evolução durante o Estágio")
st.caption(
    "Projeção química e sensorial do vinho ao longo do tempo de guarda em garrafa. "
    "As trajetórias químicas usam equações de cinética de oxidação validadas na "
    "literatura; a qualidade prevista vem de um modelo treinado nessas trajetórias."
)


@st.cache_resource
def load_aging():
    return AgingModel.load(MODELS_DIR / "aging.joblib")


model = load_aging()

wine_type = st.radio(
    "Tipo de vinho",
    options=["red", "white"],
    format_func=lambda x: "Tinto" if x == "red" else "Branco",
    horizontal=True,
    key="aging_type",
)

ranges = feature_ranges(wine_type)
defaults = default_chemistry(wine_type)

st.subheader("Química inicial (à entrada na garrafa)")
col1, col2, col3 = st.columns(3)
chem: dict[str, float] = {}
features_split = np.array_split(FEATURE_COLUMNS, 3)
for col, group in zip((col1, col2, col3), features_split):
    with col:
        for f in group:
            lo, hi, _ = ranges[f]
            step = max((hi - lo) / 200, 1e-4)
            chem[f] = st.slider(
                FEATURE_LABELS_PT[f],
                min_value=float(round(lo, 4)),
                max_value=float(round(hi, 4)),
                value=float(round(defaults[f], 4)),
                step=float(round(step, 4)),
                key=f"aging_{wine_type}_{f}",
            )

c1, c2 = st.columns([1, 2])
with c1:
    init_q = st.number_input(
        "Qualidade inicial (à prova hoje)", 3.0, 9.0, 6.0, 0.1,
        help="Pontuação sensorial atual usada como ponto de partida da curva.",
    )
    horizon = st.slider("Horizonte (anos)", 1, 20, 12)

predicted_peak = peak_year(chem, wine_type)
with c2:
    st.metric("Pico previsto (heurística enológica)", f"~ {predicted_peak:.1f} anos")
    st.caption(
        "O pico é estimado a partir de álcool, acidez fixa, SO₂ livre e acidez volátil."
    )

years_grid = np.arange(0.0, horizon + 0.5, 0.5)
curve = model.predict_curve(chem, wine_type, init_q, years_grid=years_grid)

# Quality plot
fig_q = go.Figure()
fig_q.add_trace(
    go.Scatter(
        x=curve["year"],
        y=curve["predicted_quality"],
        mode="lines+markers",
        line=dict(color="#7a1f3a", width=3),
        name="Qualidade prevista",
    )
)
fig_q.add_vline(x=predicted_peak, line_dash="dash", line_color="grey",
                annotation_text=f"Pico {predicted_peak:.1f} a", annotation_position="top")
fig_q.update_layout(
    title="Curva de qualidade durante o estágio",
    xaxis_title="Anos em garrafa",
    yaxis_title="Qualidade prevista (0–10)",
    height=400,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_q, use_container_width=True)

# Chemistry plot — pick the most informative variables
key_features = ["free sulfur dioxide", "total sulfur dioxide",
                "volatile acidity", "fixed acidity", "pH", "sulphates"]
fig_c = make_subplots(rows=2, cols=3, subplot_titles=[FEATURE_LABELS_PT[f] for f in key_features])
for i, f in enumerate(key_features):
    r, c = divmod(i, 3)
    fig_c.add_trace(
        go.Scatter(x=curve["year"], y=curve[f], mode="lines", line=dict(color="#2c3e50")),
        row=r + 1, col=c + 1,
    )
fig_c.update_layout(
    height=450,
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    title="Evolução química",
)
for i in range(6):
    fig_c.update_xaxes(title_text="Anos", row=(i // 3) + 1, col=(i % 3) + 1)
st.plotly_chart(fig_c, use_container_width=True)

with st.expander("Ver tabela completa da trajetória"):
    show_cols = ["year", "predicted_quality"] + FEATURE_COLUMNS
    rounded = curve[show_cols].copy()
    for c in rounded.columns:
        if c != "year":
            rounded[c] = rounded[c].round(3)
    st.dataframe(rounded, use_container_width=True, hide_index=True)
