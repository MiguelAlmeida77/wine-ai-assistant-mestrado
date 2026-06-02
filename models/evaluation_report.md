# Relatório de Avaliação

## Preditor de qualidade (UCI, Stratified 5-Fold)

| Tipo | Features | Modelo | R²  (μ±σ) | RMSE (μ±σ) | MAE (μ±σ) |
|------|----------|--------|-----------|------------|-----------|
| red | raw | baseline_mean | -0.000 ± 0.000 | 0.823 ± 0.008 | 0.694 ± 0.003 |
| red | raw | random_forest | 0.374 ± 0.069 | 0.650 ± 0.037 | 0.500 ± 0.024 |
| red | raw | gradient_boosting | 0.336 ± 0.079 | 0.670 ± 0.041 | 0.520 ± 0.032 |
| red | raw | xgboost | 0.347 ± 0.079 | 0.664 ± 0.041 | 0.510 ± 0.029 |
| red | engineered | baseline_mean | -0.000 ± 0.000 | 0.823 ± 0.008 | 0.694 ± 0.003 |
| red | engineered | random_forest | 0.372 ± 0.072 | 0.651 ± 0.039 | 0.501 ± 0.025 |
| red | engineered | gradient_boosting | 0.329 ± 0.075 | 0.674 ± 0.039 | 0.520 ± 0.026 |
| red | engineered | xgboost | 0.344 ± 0.074 | 0.666 ± 0.039 | 0.509 ± 0.025 |
| white | raw | baseline_mean | -0.000 ± 0.000 | 0.891 ± 0.008 | 0.679 ± 0.003 |
| white | raw | random_forest | 0.388 ± 0.023 | 0.696 ± 0.015 | 0.535 ± 0.009 |
| white | raw | gradient_boosting | 0.366 ± 0.021 | 0.709 ± 0.014 | 0.548 ± 0.009 |
| white | raw | xgboost | 0.367 ± 0.026 | 0.708 ± 0.018 | 0.541 ± 0.010 |
| white | engineered | baseline_mean | -0.000 ± 0.000 | 0.891 ± 0.008 | 0.679 ± 0.003 |
| white | engineered | random_forest | 0.387 ± 0.022 | 0.697 ± 0.015 | 0.536 ± 0.009 |
| white | engineered | gradient_boosting | 0.362 ± 0.015 | 0.711 ± 0.012 | 0.550 ± 0.007 |
| white | engineered | xgboost | 0.365 ± 0.016 | 0.709 ± 0.011 | 0.542 ± 0.006 |

## Preditor de mercado (Wine Reviews, GroupKFold por winery)

Modelo: **xgboost+tfidf** • amostras: 40,000

| Métrica | μ ± σ |
|---------|-------|
| R²      | 0.684 ± 0.011 |
| RMSE    | 1.711 ± 0.011 |
| MAE     | 1.334 ± 0.007 |