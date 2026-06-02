"""Módulo 2: Recomendação de blends por otimização restrita."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.loader import FEATURE_COLUMNS, FEATURE_LABELS_PT, load_raw
from src.models.blend_bo import pareto_front, recommend_bo
from src.models.blend_recommender import (
    BaseWine,
    BlendConstraints,
    recommend_blend,
    sample_base_wines_from_dataset,
)
from src.models.quality_predictor import MODELS_DIR, QualityModel
from src.models.uncertainty import IntervalModel
from src.utils.ranges import feature_ranges

st.set_page_config(page_title="Recomendação de Blends", layout="wide")
st.title("Recomendação de Blends")
st.caption(
    "Otimiza as proporções de vinhos-base para maximizar a qualidade prevista, "
    "respeitando restrições enológicas (álcool, acidez volátil, SO₂)."
)


@st.cache_resource
def load_models() -> dict[str, QualityModel]:
    return {
        "red": QualityModel.load(MODELS_DIR / "quality_red.joblib"),
        "white": QualityModel.load(MODELS_DIR / "quality_white.joblib"),
    }


@st.cache_resource
def load_intervals() -> dict[str, IntervalModel | None]:
    out: dict[str, IntervalModel | None] = {}
    for wt in ("red", "white"):
        p = MODELS_DIR / f"interval_{wt}.joblib"
        out[wt] = IntervalModel.load(p) if p.exists() else None
    return out


@st.cache_data
def cached_raw():
    return load_raw("both")


models = load_models()
intervals = load_intervals()
df_raw = cached_raw()

wine_type = st.radio(
    "Tipo de vinho",
    options=["red", "white"],
    format_func=lambda x: "Tinto" if x == "red" else "Branco",
    horizontal=True,
    key="blend_type",
)

# --- Base wines ---
st.subheader("Vinhos-base")

n_bases = st.slider("Número de vinhos-base", 2, 6, 4)

mode = st.radio(
    "Fonte dos vinhos-base",
    options=["Exemplos do dataset", "Edição manual"],
    horizontal=True,
)

if mode == "Exemplos do dataset":
    seed = st.number_input("Seed para reamostragem", 0, 10_000, 7, step=1)
    base_wines = sample_base_wines_from_dataset(df_raw, wine_type, n=n_bases, seed=int(seed))
else:
    ranges = feature_ranges(wine_type)
    base_wines = []
    for i in range(n_bases):
        with st.expander(f"Vinho-base {i+1}", expanded=(i < 2)):
            name = st.text_input("Nome", value=f"Base {i+1}", key=f"name_{i}")
            cols = st.columns(3)
            chem: dict[str, float] = {}
            features_split = np.array_split(FEATURE_COLUMNS, 3)
            for col, group in zip(cols, features_split):
                with col:
                    for f in group:
                        lo, hi, med = ranges[f]
                        chem[f] = st.number_input(
                            FEATURE_LABELS_PT[f],
                            value=float(round(med, 3)),
                            min_value=float(round(lo, 3)),
                            max_value=float(round(hi, 3)),
                            key=f"manual_{i}_{f}",
                        )
            base_wines.append(BaseWine(name=name, chemistry=chem))

# Show base table
df_bases = pd.DataFrame(
    [{"vinho": b.name, **{FEATURE_LABELS_PT[c]: b.chemistry[c] for c in FEATURE_COLUMNS}} for b in base_wines]
)
with st.expander("Ver química dos vinhos-base"):
    st.dataframe(df_bases, use_container_width=True, hide_index=True)

# --- Constraints ---
st.subheader("Restrições do blend final")
c1, c2, c3, c4 = st.columns(4)
with c1:
    min_alc = st.number_input("Álcool mínimo (% vol.)", 8.0, 15.0, 11.0, 0.1)
with c2:
    max_alc = st.number_input("Álcool máximo (% vol.)", 9.0, 16.0, 14.5, 0.1)
with c3:
    max_va = st.number_input("Acidez volátil máxima (g/L)", 0.2, 1.5, 0.9, 0.05)
with c4:
    max_so2 = st.number_input("SO₂ total máximo (mg/L)", 50.0, 350.0, 200.0, 5.0)

target_style = st.checkbox("Alvejar perfil-estilo específico (avançado)")
target_profile = None
target_weight = 0.0
if target_style:
    target_weight = st.slider("Peso do alvo vs qualidade", 0.0, 2.0, 0.5, 0.1)
    ranges = feature_ranges(wine_type)
    cols = st.columns(3)
    target_profile = {}
    features_split = np.array_split(FEATURE_COLUMNS, 3)
    for col, group in zip(cols, features_split):
        with col:
            for f in group:
                lo, hi, med = ranges[f]
                target_profile[f] = st.number_input(
                    f"Alvo: {FEATURE_LABELS_PT[f]}",
                    value=float(round(med, 3)),
                    min_value=float(round(lo, 3)),
                    max_value=float(round(hi, 3)),
                    key=f"target_{f}",
                )

constraints = BlendConstraints(
    min_alcohol=min_alc,
    max_alcohol=max_alc,
    max_volatile_acidity=max_va,
    max_total_so2=max_so2,
    target_profile=target_profile,
    target_weight=target_weight,
)

st.subheader("Método de otimização")
optimizer = st.radio(
    "Otimizador",
    options=["SLSQP (rápido, local)", "Bayesian / TPE (exploração global)"],
    horizontal=True,
    help=(
        "SLSQP converge em milissegundos mas pode ficar preso em ótimos locais. "
        "BO/TPE explora melhor a superfície mas demora alguns segundos."
    ),
)
n_trials_bo = 80
if optimizer.startswith("Bayesian"):
    n_trials_bo = st.slider("Trials BO", 30, 250, 100, 10)

# --- Optimize ---
if st.button("Calcular blend ótimo", type="primary", use_container_width=True):
    with st.spinner("A otimizar..."):
        if optimizer.startswith("SLSQP"):
            result = recommend_blend(base_wines, models[wine_type], constraints=constraints)
        else:
            bo = recommend_bo(
                base_wines,
                models[wine_type],
                constraints=constraints,
                interval_model=intervals.get(wine_type),
                n_trials=n_trials_bo,
            )
            # Adapt BOBlendResult to the same display contract used below
            from src.models.blend_recommender import BlendResult
            result = BlendResult(
                proportions=bo.proportions,
                blended_chemistry=bo.blended_chemistry,
                predicted_quality=bo.predicted_quality,
                converged=True,
                message=(
                    f"BO ok. Intervalo 80% conformal: "
                    f"[{bo.quality_lo:.2f}, {bo.quality_hi:.2f}]"
                    if bo.quality_lo is not None
                    else "BO ok."
                ),
            )

    if not result.converged:
        st.warning(f"Otimização não convergiu: {result.message}")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.metric("Qualidade prevista do blend", f"{result.predicted_quality:.2f} / 10")
        prop_df = pd.DataFrame(
            [{"vinho": k, "percentagem": v * 100} for k, v in result.proportions.items()]
        ).sort_values("percentagem", ascending=False)
        fig = px.pie(prop_df, values="percentagem", names="vinho", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Reds if wine_type == "red" else px.colors.sequential.YlOrBr)
        fig.update_layout(title="Proporções recomendadas", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Química do blend resultante**")
        chem_df = pd.DataFrame(
            [{"variável": FEATURE_LABELS_PT[c], "valor": round(v, 3)} for c, v in result.blended_chemistry.items()]
        )
        st.dataframe(chem_df, use_container_width=True, hide_index=True, height=420)

    if optimizer.startswith("Bayesian"):
        st.caption(result.message)

# --- Pareto front ---
st.divider()
st.subheader("Exploração Pareto (qualidade × álcool-alvo)")
st.caption(
    "Quando há objetivos em conflito (ex.: maximizar qualidade *e* aproximar o "
    "álcool de um valor-alvo), o Pareto front mostra as combinações onde não é "
    "possível melhorar um objetivo sem piorar o outro. NSGA-II via Optuna."
)
pareto_c1, pareto_c2 = st.columns(2)
with pareto_c1:
    target_alc = st.number_input("Álcool-alvo (% vol.)", 9.0, 16.0, 12.5, 0.1)
with pareto_c2:
    n_trials_pareto = st.slider("Trials Pareto", 60, 300, 150, 10)

if st.button("Gerar Pareto front"):
    with st.spinner("A explorar..."):
        pf = pareto_front(
            base_wines,
            models[wine_type],
            target_alcohol=target_alc,
            n_trials=n_trials_pareto,
        )
    if pf.empty:
        st.warning("Nenhum ponto Pareto encontrado — tente mais trials.")
    else:
        st.markdown(f"**{len(pf)} soluções na fronteira de Pareto**")
        fig = px.scatter(
            pf, x="álcool", y="qualidade",
            hover_data=[c for c in pf.columns if c.startswith("%_")] + ["acidez_volátil", "SO2_total"],
            title=f"Pareto: qualidade vs. distância ao álcool-alvo ({target_alc:.1f})",
            color="dist_alcool", color_continuous_scale="Reds_r",
        )
        fig.add_vline(x=target_alc, line_dash="dash", line_color="grey",
                      annotation_text=f"Alvo {target_alc:.1f}", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pf, use_container_width=True, hide_index=True)
