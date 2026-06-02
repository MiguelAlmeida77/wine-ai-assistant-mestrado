"""Módulo 1: Predição de qualidade sensorial a partir da química."""
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

from src.data.loader import FEATURE_COLUMNS, FEATURE_LABELS_PT
from src.models.counterfactual import find_counterfactual
from src.models.quality_predictor import MODELS_DIR, QualityModel
from src.models.uncertainty import IntervalModel
from src.utils.ranges import default_chemistry, feature_ranges

st.set_page_config(page_title="Predição de Qualidade", layout="wide")
st.title("Predição de Qualidade Sensorial")
st.caption(
    "Estima a pontuação sensorial (0–10) a partir do perfil físico-químico, "
    "usando um modelo XGBoost treinado em ~6 500 vinhos do dataset UCI Wine Quality."
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
        path = MODELS_DIR / f"interval_{wt}.joblib"
        out[wt] = IntervalModel.load(path) if path.exists() else None
    return out


models = load_models()
intervals = load_intervals()

wine_type = st.radio(
    "Tipo de vinho",
    options=["red", "white"],
    format_func=lambda x: "Tinto" if x == "red" else "Branco",
    horizontal=True,
)

ranges = feature_ranges(wine_type)
defaults = default_chemistry(wine_type)

st.subheader("Perfil químico")
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
                key=f"{wine_type}_{f}",
            )

st.divider()

model = models[wine_type]
prediction = float(model.predict(chem)[0])

interval = intervals.get(wine_type)
interval_pred = interval.predict(chem) if interval is not None else None

c1, c2 = st.columns([1, 2])
with c1:
    if interval_pred is not None:
        median = float(interval_pred["median"][0])
        conf_lo = float(interval_pred["conformal_lo"][0])
        conf_hi = float(interval_pred["conformal_hi"][0])
        q_lo = float(interval_pred["quantile_lo"][0])
        q_hi = float(interval_pred["quantile_hi"][0])
        st.metric(
            "Qualidade prevista (mediana)",
            f"{median:.2f} / 10",
            delta=f"vs. modelo único: {prediction - median:+.2f}",
            help="Mediana do modelo de quantis. Escala UCI: 0 = inaceitável, 10 = excelente.",
        )
        st.markdown(
            f"**Intervalo 80%**\n"
            f"- *Conformal* (garantia distribuição-livre): "
            f"[{conf_lo:.2f}, {conf_hi:.2f}]\n"
            f"- *Quantile XGB* (modelo-aware): "
            f"[{q_lo:.2f}, {q_hi:.2f}]"
        )
        if interval is not None:
            cov = interval.metrics["conformal"]["empirical_coverage"]
            st.caption(f"Cobertura empírica (conformal, alvo 80%): **{cov:.1%}**")
    else:
        st.metric(
            "Qualidade prevista",
            f"{prediction:.2f} / 10",
            help="Escala UCI: 0 = inaceitável, 10 = excelente.",
        )
        st.info("Intervalos não disponíveis. Corra `scripts\\train_uncertainty.py`.")

    metrics = model.metrics[model.metrics["winner"]]
    st.caption(
        f"Modelo pontual: **{model.metrics['winner']}** • "
        f"R² teste = {metrics['r2']:.3f} • RMSE = {metrics['rmse']:.3f}"
    )

with c2:
    # Feature contribution via permutation/feature_importances
    importances = getattr(model.estimator, "feature_importances_", None)
    if importances is not None:
        imp_df = (
            pd.DataFrame({"feature": model.feature_cols, "importance": importances})
            .sort_values("importance", ascending=True)
            .tail(10)
        )
        # Translate names where possible
        imp_df["label"] = imp_df["feature"].map(
            lambda f: FEATURE_LABELS_PT.get(f, f)
        )
        fig = go.Figure(
            go.Bar(
                x=imp_df["importance"],
                y=imp_df["label"],
                orientation="h",
                marker_color="#7a1f3a",
            )
        )
        fig.update_layout(
            title="Importância das variáveis (modelo)",
            xaxis_title="Importância relativa",
            margin=dict(l=10, r=10, t=40, b=10),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("Contrafactual: o que mudar para atingir uma qualidade-alvo?"):
    st.caption(
        "Encontra a perturbação química mínima que faz a qualidade prevista atingir "
        "o alvo, respeitando os limites do dataset UCI e penalizando alterações em "
        "variáveis não-acionáveis (densidade, pH)."
    )
    pred_now = float(model.predict(chem)[0])
    suggested_target = float(round(min(pred_now + 0.5, 8.0), 1))
    target_q = st.slider(
        "Qualidade-alvo", 3.0, 8.5, suggested_target, 0.1, key=f"cf_target_{wine_type}",
        help=f"Qualidade prevista atual: {pred_now:.2f}",
    )
    if st.button("Procurar contrafactual", key=f"cf_btn_{wine_type}"):
        with st.spinner("A otimizar..."):
            cf = find_counterfactual(chem, target_q, model, wine_type)
        if not cf.feasible:
            st.warning(
                f"Não foi encontrada uma química viável dentro dos limites UCI "
                f"para atingir {target_q:.1f}. Tente um alvo mais próximo do "
                f"atual ({pred_now:.2f})."
            )
        else:
            st.success(
                f"Atingível: química ajustada → qualidade prevista {cf.achieved_quality:.2f}"
            )
            df_cf = cf.to_frame()
            st.dataframe(df_cf, use_container_width=True, hide_index=True)
            st.caption(
                "Variáveis ordenadas por impacto. *Δ* é a alteração proposta; "
                "*dentro da gama UCI* sinaliza se o valor final ainda fica dentro "
                "do percentil 1–99 do dataset original."
            )

with st.expander("Explicação local (SHAP)"):
    st.caption(
        "Mostra como cada variável contribuiu para esta predição específica, "
        "relativamente à média do dataset. Valores positivos puxam a qualidade "
        "para cima."
    )
    try:
        import shap
        from src.data.preprocessor import add_engineered_features as _aef
        sample_df = _aef(pd.DataFrame([chem]))[model.feature_cols]
        explainer = shap.TreeExplainer(model.estimator)
        shap_values = explainer.shap_values(sample_df)
        shap_arr = np.asarray(shap_values).flatten()
        contribs = pd.DataFrame({
            "feature": model.feature_cols,
            "label": [FEATURE_LABELS_PT.get(f, f) for f in model.feature_cols],
            "shap": shap_arr,
            "valor": sample_df.iloc[0].values,
        }).sort_values("shap", key=lambda s: s.abs(), ascending=True).tail(11)
        fig_shap = go.Figure(go.Bar(
            x=contribs["shap"], y=contribs["label"],
            orientation="h",
            marker_color=["#7a1f3a" if v > 0 else "#bbb" for v in contribs["shap"]],
            text=[f"{v:+.3f}" for v in contribs["shap"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title="Contribuição por variável (SHAP)",
            xaxis_title="Impacto na qualidade prevista",
            height=400, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            f"Valor base do modelo: {float(explainer.expected_value):.2f} • "
            f"Predição = base + Σ SHAP = {float(explainer.expected_value) + shap_arr.sum():.2f}"
        )
    except Exception as e:
        st.warning(f"SHAP indisponível: {e}")

with st.expander("Ver tabela detalhada do perfil"):
    df_show = pd.DataFrame(
        [{"variável": FEATURE_LABELS_PT[f], "valor": v} for f, v in chem.items()]
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)
