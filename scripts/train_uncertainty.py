"""Treina modelos de intervalo (quantile XGB + conformal) por tipo de vinho."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw
from src.models.quality_predictor import MODELS_DIR
from src.models.uncertainty import train_interval


def main() -> None:
    df = load_raw("both")
    summary = {}
    for wt in ("red", "white"):
        print(f"[{wt}] a treinar intervalos...")
        m = train_interval(df, wt)
        m.save(MODELS_DIR / f"interval_{wt}.joblib")
        summary[wt] = m.metrics
        cov_q = m.metrics["quantile_xgb"]["empirical_coverage"]
        cov_c = m.metrics["conformal"]["empirical_coverage"]
        w_q = m.metrics["quantile_xgb"]["mean_width"]
        w_c = m.metrics["conformal"]["mean_width"]
        print(f"  alvo={1-m.alpha:.0%} | quantile cov={cov_q:.2%} (w={w_q:.2f})"
              f" | conformal cov={cov_c:.2%} (w={w_c:.2f})")
    (MODELS_DIR / "interval_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
