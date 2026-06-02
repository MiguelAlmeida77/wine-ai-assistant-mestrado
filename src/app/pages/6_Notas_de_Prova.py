"""Módulo 6: Gerador de notas de prova a partir de metadata do vinho."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.loader import PROCESSED_DIR
from src.models.quality_predictor import MODELS_DIR
from src.models.tasting_notes import (
    DESCRIPTOR_FAMILIES,
    NotesModel,
    render_note,
)

st.set_page_config(page_title="Notas de Prova", layout="wide")
st.title("Gerador de Notas de Prova")
st.caption(
    "Modelo treinado em ~40 000 reviews da Wine Enthusiast. Extrai descritores "
    "aromáticos do texto (léxico curado de 40 termos), aprende a probabilidade "
    "marginal por descritor a partir de metadata, e devolve uma nota legível."
)


@st.cache_resource
def load_notes_model() -> NotesModel:
    return NotesModel.load(MODELS_DIR / "notes.joblib")


@st.cache_data
def choices() -> dict:
    path = PROCESSED_DIR / "choices.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from src.data.reviews_loader import load_reviews
    df = load_reviews()
    return {
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "varieties": df["variety"].value_counts().head(80).index.tolist(),
        "provinces": df["province"].dropna().value_counts().head(120).index.tolist(),
        "tasters": sorted(df["taster_name"].dropna().unique().tolist()),
        "price_median": float(df["price"].median()),
    }


model = load_notes_model()
opts = choices()

st.subheader("Identificação do vinho")
c1, c2 = st.columns(2)
with c1:
    country = st.selectbox(
        "País", options=opts["countries"],
        index=opts["countries"].index("Portugal") if "Portugal" in opts["countries"] else 0,
    )
    variety = st.selectbox("Casta / blend", options=opts["varieties"])
with c2:
    province = st.selectbox("Província / região", options=["unknown"] + opts["provinces"])
    price = st.number_input("Preço (USD)", 4.0, 2000.0, float(opts["price_median"]), 1.0)
taster = st.selectbox("Provador (opcional)", options=["unknown"] + opts["tasters"])

st.subheader("Sensibilidade da geração")
c1, c2 = st.columns(2)
with c1:
    threshold = st.slider("Limiar absoluto", 0.05, 0.50, 0.12, 0.01,
                          help="Probabilidade mínima para incluir um descritor")
with c2:
    lift_th = st.slider("Lift mínimo sobre base", 1.0, 3.0, 1.4, 0.1,
                        help="Inclui descritores raros se a probabilidade for >= lift × frequência base")

if st.button("Gerar nota de prova", type="primary", use_container_width=True):
    sample = {
        "country": country, "variety": variety,
        "province": province if province != "unknown" else "unknown",
        "taster_name": taster if taster != "unknown" else "unknown",
        "price": price,
    }
    with st.spinner("A prever descritores..."):
        probs = model.predict_probs(sample)

    base_freq = model.metrics.get("frequencies", {})
    note = render_note(probs, threshold=threshold, lift_threshold=lift_th,
                       base_frequencies=base_freq)

    st.markdown("### Nota gerada")
    st.info(note)

    st.markdown("### Detalhe por descritor")
    c1, c2 = st.columns([1, 1])
    with c1:
        # Top-15 por probabilidade absoluta
        top = probs.sort_values(ascending=False).head(15).reset_index()
        top.columns = ["descritor", "probabilidade"]
        fig = px.bar(top, x="probabilidade", y="descritor", orientation="h",
                     color="probabilidade", color_continuous_scale="Reds")
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"},
                          title="Top-15 descritores")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Por família
        rows = []
        for fam, members in DESCRIPTOR_FAMILIES.items():
            for d in members:
                if d in probs.index:
                    rows.append({
                        "família": fam, "descritor": d,
                        "probabilidade": round(float(probs[d]), 3),
                        "base": round(float(base_freq.get(d, 0)), 3),
                    })
        df = pd.DataFrame(rows).sort_values(["família", "probabilidade"], ascending=[True, False])
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)

    st.caption(
        f"Modelo F1 micro = {model.metrics['f1_micro']:.2f} • "
        f"F1 macro = {model.metrics['f1_macro']:.2f} (multi-label, GroupKFold por winery)"
    )
