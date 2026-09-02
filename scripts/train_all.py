"""Treina todos os modelos do protótipo end-to-end.

Uso (a partir da raiz do projeto):
    .\\.venv\\Scripts\\python.exe scripts\\train_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw, save_processed
from src.data.reviews_loader import load_reviews, save_processed_reviews
from src.models.market_predictor import train as train_market
from src.models.quality_predictor import MODELS_DIR, train_all as train_quality
from src.models.tasting_notes import train as train_notes
from src.models.uncertainty import train_interval


def main() -> None:
    print("[1/5] A carregar dados UCI...")
    df = load_raw("both")
    out = save_processed(df)
    print(f"      {len(df)} amostras guardadas em {out}")

    print("[2/5] A treinar modelos de qualidade (tinto + branco)...")
    qmodels = train_quality()
    for wt, m in qmodels.items():
        win = m.metrics["winner"]
        r2 = m.metrics[win]["r2"]
        print(f"      {wt}: {win} R²={r2:.3f}")

    print("[3/5] A treinar modelos de intervalo (quantile + conformal)...")
    import json
    iv_summary = {}
    for wt in ("red", "white"):
        m = train_interval(df, wt)
        m.save(MODELS_DIR / f"interval_{wt}.joblib")
        iv_summary[wt] = m.metrics
        c = m.metrics["conformal"]["empirical_coverage"]
        print(f"      {wt}: conformal coverage={c:.1%} (alvo {1-m.alpha:.0%})")
    (MODELS_DIR / "interval_metrics.json").write_text(
        json.dumps(iv_summary, indent=2), encoding="utf-8"
    )

    print("[4/5] A treinar modelo de mercado (Wine Reviews)...")
    df_rev = load_reviews()
    save_processed_reviews(df_rev)
    mkt = train_market()
    print(f"      R²={mkt.metrics['r2']:.3f} • RMSE={mkt.metrics['rmse']:.2f}")

    print("[5/5] A treinar gerador de notas de prova...")
    notes = train_notes()
    print(f"      F1 micro={notes.metrics['f1_micro']:.3f} • {notes.metrics['n_descriptors']} descritores")

    print("\nConcluído. Lance a UI com:")
    print(r"  .\.venv\Scripts\streamlit.exe run src\app\app.py")


if __name__ == "__main__":
    main()
