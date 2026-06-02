# Assistente Inteligente para Afinação, Estágio e Blend de Vinhos

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

Protótipo desenvolvido no âmbito de dissertação de mestrado. Sistema de apoio
à decisão para enólogos de adegas pequenas e médias, baseado em Inteligência
Artificial. Cobre os quatro objetivos definidos na proposta de tese — e
adiciona aceitação de mercado, quantificação de incerteza e *tuning*
sistemático:

1. **Modelo preditivo** — química → pontuação sensorial (XGB / RF, comparados
   por *ablation* estratificado).
2. **Recomendador de blends** — otimização restrita (SLSQP multi-start) sobre
   o modelo de qualidade, com restrições enológicas (álcool, acidez volátil,
   SO₂, perfil-alvo).
3. **Previsão de envelhecimento** — trajetórias químicas analíticas calibradas
   pela literatura + GradientBoosting para a curva de qualidade.
4. **Interface interativa** — aplicação Streamlit multi-página em português.
5. **Aceitação de mercado** — XGBoost + TF-IDF treinado em 120k *reviews*
   Wine Enthusiast; *Group K-Fold* por *winery* (sem leakage entre produtores).
6. **Quantificação de incerteza** — quantis XGB + *split-conformal*; intervalos
   80% com cobertura empírica reportada.
7. **Tuning** — Optuna (TPE) para todos os modelos, *best_params* persistidos.
8. **Contrafactuais** — dado um vinho com qualidade prevista X, devolve a
   *menor* alteração química que atinge um alvo Y, dentro dos limites do
   *dataset* (SLSQP multi-start com fast-path numpy).
9. **Otimização bayesiana de blends** — Optuna TPE como alternativa global ao
   SLSQP; NSGA-II multi-objetivo gera *Pareto front* qualidade × álcool-alvo.
10. **Augmentação varietal sintética** — perturbações por casta (Touriga,
    Alvarinho, Loureiro, Vinhão, etc.) calibradas pela literatura enológica.
    Comparação com/sem augmentação reportada (GroupKFold para evitar leakage:
    sem ganho real de generalização — resultado academicamente honesto).
11. **Geração de notas de prova** — léxico de 40 descritores extraídos do
    Wine Reviews; multi-output logístico (variedade/país/preço → probabilidade
    por descritor); renderização em PT por templates.

## Stack

- Python 3.14 · numpy · pandas · scikit-learn 1.8 · XGBoost 3.2
- scipy (otimização) · SHAP (explicabilidade)
- Streamlit 1.57 · Plotly

## Estrutura

```
wine-ai-assistant/
├── data/
│   ├── raw/               # CSVs UCI Wine Quality + Wine Reviews
│   ├── processed/         # parquet limpo
│   └── synthetic/         # trajetórias de envelhecimento
├── models/                # modelos persistidos (joblib) + métricas (json) + relatórios
├── src/
│   ├── data/
│   │   ├── loader.py            # UCI Wine Quality
│   │   ├── preprocessor.py
│   │   ├── synthetic_aging.py   # equações de cinética enológica
│   │   └── reviews_loader.py    # Wine Reviews (Wine Enthusiast)
│   ├── models/
│   │   ├── quality_predictor.py # XGB/RF por tipo, comparados
│   │   ├── blend_recommender.py # SLSQP multi-start com restrições
│   │   ├── aging_predictor.py   # GBM em trajetórias sintéticas
│   │   ├── uncertainty.py       # quantile XGB + split-conformal
│   │   ├── market_predictor.py  # XGB + TF-IDF
│   │   └── evaluation.py        # CV estratificada/agrupada + ablation
│   ├── utils/             # ranges para sliders
│   └── app/               # Streamlit multi-página
│       └── pages/         # Predição, Blends, Envelhecimento, Sobre, Mercado
├── scripts/
│   ├── setup.ps1
│   ├── train_all.py             # pipeline completo
│   ├── train_uncertainty.py     # só os intervalos
│   └── tune.py                  # Optuna
├── tests/                  # pytest suite
└── requirements.txt
```

## Deploy público (Streamlit Community Cloud)

Este repositório está pronto para deploy gratuito em
[share.streamlit.io](https://share.streamlit.io). Configuração:

| Campo | Valor |
|---|---|
| Repository | `MiguelAlmeida77/wine-ai-assistant-mestrado` |
| Branch | `main` |
| Main file path | `src/app/app.py` |
| Python version | `3.12` (lido de `runtime.txt`) |

O primeiro build instala todas as dependências (≈ 5 min). Os modelos já estão
no repositório (12 MB total comprimidos), e o dataset Wine Reviews tem assets
leves pré-computados em `data/processed/` (305 KB), evitando o download do
CSV de 50 MB no servidor.

## Instalação rápida (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Ou passo a passo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\train_all.py
.\.venv\Scripts\streamlit.exe run src\app\app.py
```

A UI abre em `http://localhost:8501`.

## Dados

- **UCI Wine Quality** (Cortez et al., 2009): 1599 tintos + 4898 brancos da
  região do Vinho Verde, com 11 variáveis físico-químicas e pontuação
  sensorial média de 3 provadores (0–10).
- **Wine Reviews** (Wine Enthusiast, ~130k reviews, mirror TidyTuesday):
  país, casta, província, preço, provador, descritor textual e pontos
  80–100 do crítico. Usado para o modelo de aceitação de mercado.

Os CSVs são descarregados de mirrors do GitHub (o servidor original da UCI
apresenta handshake TLS inconsistente em algumas configurações Windows).

## Métricas (após `train_all.py` + ablation)

**Qualidade — Stratified 5-Fold (UCI)**
- Tinto: RandomForest R² 0.37 ± 0.07 • RMSE 0.65 ± 0.04
- Branco: RandomForest R² 0.39 ± 0.02 • RMSE 0.70 ± 0.02
- (XGBoost e GradientBoosting ficam ≈1 ponto percentual abaixo da RF neste tamanho de dataset.)

**Incerteza — intervalos 80%**
- Tinto: cobertura conformal 82.8% (largura 1.63), quantile XGB 70.6% (1.12)
- Branco: cobertura conformal 80.7% (largura 1.91), quantile XGB 63.5% (1.36)
- → conformal cumpre a garantia teórica, quantile XGB sub-cobre.

**Aceitação de mercado — GroupKFold por winery (Wine Reviews, 120k linhas)**
- R² 0.68 ± 0.01 • RMSE 1.71 ± 0.01 pontos (escala 80–100)

**Envelhecimento (trajetórias sintéticas + GradientBoosting)**
- R² 0.41 • RMSE 0.79 (em curvas sintéticas calibradas pela literatura)

Detalhes em `models/evaluation_report.md`.

## Limitações documentadas

- Cobertura regional limitada (Vinho Verde) — generalização exige re-treino.
- A qualidade é proxy: média de provadores, não painel sensorial dedicado.
- Trajetórias de estágio são sintéticas; validação industrial requer dataset
  longitudinal real.
- Mistura química do blend assume linearidade em volume (exata para álcool,
  açúcar e SO₂; aproximada para pH e densidade).

## Próximos passos (para fora deste ciclo)

- Dados sensoriais reais de adega parceira.
- Descritores aromáticos (Kaggle Wine Reviews) num modelo multimodal.
- Validação longitudinal do modelo de envelhecimento.
