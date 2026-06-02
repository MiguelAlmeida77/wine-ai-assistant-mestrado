"""Módulo 5: Aceitação de mercado (modelo treinado em Wine Reviews)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.reviews_loader import load_reviews
from src.models.market_predictor import MarketModel, composite_score
from src.models.quality_predictor import MODELS_DIR

st.set_page_config(page_title="Aceitação de Mercado", layout="wide")
st.title("Aceitação de Mercado")
st.caption(
    "Modelo treinado em ~120 000 reviews da Wine Enthusiast (TF-IDF nas notas "
    "de prova + país, casta, província, preço, provador). Devolve a pontuação "
    "esperada do crítico (escala 80–100)."
)


@st.cache_resource
def load_market_model() -> MarketModel:
    return MarketModel.load(MODELS_DIR / "market.joblib")


@st.cache_data
def load_choices() -> dict:
    df = load_reviews()
    return {
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "varieties": (
            df["variety"].value_counts().head(80).index.tolist()
        ),
        "provinces": (
            df["province"].dropna().value_counts().head(120).index.tolist()
        ),
        "tasters": sorted(df["taster_name"].dropna().unique().tolist()),
        "price_median": float(df["price"].median()),
    }


model = load_market_model()
choices = load_choices()

st.subheader("Identificação do vinho")
c1, c2 = st.columns(2)
with c1:
    country = st.selectbox(
        "País", options=choices["countries"],
        index=choices["countries"].index("Portugal") if "Portugal" in choices["countries"] else 0,
    )
    variety = st.selectbox("Casta / blend", options=choices["varieties"])
with c2:
    province = st.selectbox(
        "Província / região", options=["unknown"] + choices["provinces"]
    )
    price = st.number_input(
        "Preço (USD)", min_value=4.0, max_value=2000.0,
        value=float(choices["price_median"]), step=1.0,
    )

taster = st.selectbox(
    "Provador (opcional)",
    options=["unknown"] + choices["tasters"],
    help="Diferentes provadores têm diferentes calibrações — útil para simular o efeito.",
)

st.subheader("Nota de prova (descritores)")
desc = st.text_area(
    "Descreva o vinho (português ou inglês — o modelo foi treinado em inglês, "
    "termos enológicos similares funcionam):",
    value="aromas of dark cherry, vanilla and tobacco. Firm tannins with bright acidity and a long mineral finish.",
    height=120,
)

if st.button("Prever pontuação de crítico", type="primary", use_container_width=True):
    sample = {
        "country": country,
        "variety": variety,
        "province": province if province != "unknown" else "unknown",
        "taster_name": taster if taster != "unknown" else "unknown",
        "price": price,
        "description": desc,
    }
    pred = float(model.predict(sample)[0])

    c1, c2 = st.columns([1, 2])
    with c1:
        # Wine Enthusiast classifications (public)
        if pred >= 98:
            label = "Classic — vinho icónico"
        elif pred >= 94:
            label = "Superb — excecional"
        elif pred >= 90:
            label = "Excellent — muito recomendado"
        elif pred >= 87:
            label = "Very Good — recomendado"
        elif pred >= 83:
            label = "Good — sólido / mesa"
        else:
            label = "Acceptable — comercial"
        st.metric("Pontuação esperada", f"{pred:.1f} / 100")
        st.caption(label)

        m = model.metrics
        st.caption(
            f"Modelo: XGBoost + TF-IDF • R² teste = {m['r2']:.3f} • "
            f"RMSE = {m['rmse']:.2f}"
        )

    with c2:
        st.markdown("**Score compósito (qualidade técnica × aceitação de mercado)**")
        q = st.slider(
            "Qualidade técnica (do modelo de química, 0–10)",
            0.0, 10.0, 6.0, 0.1, key="composite_q",
        )
        w = st.slider("Peso do mercado", 0.0, 1.0, 0.4, 0.05, key="composite_w")
        composite = composite_score(q, pred, weight_market=w)
        st.metric("Score compósito (0–10)", f"{composite:.2f}")
        st.caption(
            "O mercado é re-escalado de 80–100 → 0–10. Útil para o enólogo "
            "balancear excelência técnica com posicionamento comercial."
        )

st.divider()
st.subheader("Distribuição do preço vs. pontos no dataset (orientação)")
df_choice = load_reviews().sample(n=8000, random_state=0)
fig = px.scatter(
    df_choice, x="price", y="points", color="country",
    opacity=0.4, log_x=True,
    title="Preço (USD, log) vs. pontos — amostra de 8 000 reviews",
)
fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)
