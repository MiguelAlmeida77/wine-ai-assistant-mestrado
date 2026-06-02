"""Pré-computa pequenos ficheiros para a UI evitar carregar o CSV de 50 MB.

Saída (em `data/processed/`):
  - choices.json: listas de país/casta/província/provador e preço mediano
  - reviews_sample.parquet: 8 000 linhas para os gráficos de scatter

Estes ficheiros são committed ao repo; a app no Streamlit Cloud carrega-os
em segundos sem precisar do CSV completo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import PROCESSED_DIR
from src.data.reviews_loader import load_reviews


def main() -> None:
    print("A carregar Wine Reviews...")
    df = load_reviews()
    print(f"  {len(df):,} linhas")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    choices = {
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "varieties": df["variety"].value_counts().head(80).index.tolist(),
        "provinces": df["province"].dropna().value_counts().head(120).index.tolist(),
        "tasters": sorted(df["taster_name"].dropna().unique().tolist()),
        "price_median": float(df["price"].median()),
    }
    out_choices = PROCESSED_DIR / "choices.json"
    out_choices.write_text(
        json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  choices.json: {out_choices.stat().st_size/1024:.1f} KB")

    sample = df.sample(n=min(8000, len(df)), random_state=0)[
        ["country", "variety", "province", "price", "points", "title"]
    ]
    out_sample = PROCESSED_DIR / "reviews_sample.parquet"
    sample.to_parquet(out_sample, index=False)
    print(f"  reviews_sample.parquet: {out_sample.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
