"""Página inicial do Assistente Inteligente de Enologia."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/app/app.py` from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Assistente Enológico IA",
    page_icon="W",
    layout="wide",
)

st.title("Assistente Inteligente para Afinação e Blend de Vinhos")
st.caption(
    "Protótipo desenvolvido no âmbito da dissertação de mestrado — apoio à decisão "
    "para enólogos de adegas pequenas e médias."
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### Objetivo
        Disponibilizar a enólogos uma ferramenta baseada em Inteligência Artificial
        para apoiar decisões técnicas críticas:

        1. **Predição de qualidade** — estimar a pontuação sensorial provável a partir
           do perfil físico-químico de um vinho, com intervalo de incerteza e
           explicações.
        2. **Recomendação de blends** — otimizar as proporções de vinhos-base
           disponíveis para maximizar qualidade prevista, respeitando restrições
           técnicas e regulamentares.
        3. **Aceitação de mercado** — estimar a pontuação esperada de crítica a
           partir de metadados e descritores do vinho.
        4. **Notas de prova** — gerar automaticamente descritores aromáticos e
           sensoriais característicos do vinho.

        Use o menu lateral para aceder a cada módulo.
        """
    )

with col2:
    st.info(
        "**Dados**\n\n"
        "UCI Wine Quality (Cortez et al., 2009)\n\n"
        "• 1 599 vinhos tintos\n"
        "• 4 898 vinhos brancos\n"
        "• Variedade *Vinho Verde* (Portugal)\n"
        "• 11 atributos químicos + nota sensorial 0–10",
        icon="ℹ️",
    )

st.divider()

st.subheader("Como interpretar os resultados")
st.markdown(
    """
    - As pontuações de qualidade são previsões probabilísticas, não substituem o
      painel de provadores. Devem ser usadas como ranking entre alternativas.
    - A mistura química do blend assume linearidade em volume — válido para
      álcool, açúcar e SO₂; aproximado para pH e densidade.
    """
)

st.caption("© 2026 — Dissertação de Mestrado.")
