"""Demo do gerador de notas para varios vinhos."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quality_predictor import MODELS_DIR
from src.models.tasting_notes import NotesModel, render_note


def main() -> None:
    m = NotesModel.load(MODELS_DIR / "notes.joblib")
    base = m.metrics.get("frequencies")
    samples = [
        {"country": "Portugal", "variety": "Touriga Nacional",
         "province": "Douro", "taster_name": "Roger Voss", "price": 35.0},
        {"country": "Portugal", "variety": "Alvarinho",
         "province": "Vinho Verde", "taster_name": "Roger Voss", "price": 18.0},
        {"country": "France", "variety": "Pinot Noir",
         "province": "Burgundy", "taster_name": "Roger Voss", "price": 80.0},
        {"country": "Italy", "variety": "Nebbiolo",
         "province": "Piedmont", "taster_name": "Kerin O'Keefe", "price": 60.0},
    ]
    for s in samples:
        probs = m.predict_probs(s)
        note = render_note(probs, threshold=0.12, lift_threshold=1.4,
                            base_frequencies=base)
        print(f"\n{s['variety']} ({s['country']}, {s['province']}, USD {s['price']:.0f}):")
        print(" ", note)


if __name__ == "__main__":
    main()
