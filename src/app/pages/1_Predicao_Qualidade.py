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
from src.models.quality_predictor import MODELS_DIR, QualityModel
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


models = load_models()

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

c1, c2 = st.columns([1, 2])
with c1:
    st.metric(
        "Qualidade prevista",
        f"{prediction:.2f} / 10",
        help="Escala UCI: 0 = inaceitável, 10 = excelente. A maioria dos vinhos cai em 4–7.",
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
