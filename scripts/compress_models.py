"""Re-comprime os ficheiros joblib em `models/` com nível 3.

Reduz tipicamente 50–70 % com penalização desprezável no tempo de load.
Necessário para caber confortavelmente no GitHub e no Streamlit Cloud.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Touch imports so unpickle finds the classes
from src.models.market_predictor import MarketModel  # noqa: F401
from src.models.quality_predictor import MODELS_DIR, QualityModel  # noqa: F401
from src.models.tasting_notes import NotesModel  # noqa: F401
from src.models.uncertainty import IntervalModel  # noqa: F401


def main() -> None:
    before = after = 0
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        size0 = p.stat().st_size
        obj = joblib.load(p)
        joblib.dump(obj, p, compress=3)
        size1 = p.stat().st_size
        before += size0
        after += size1
        ratio = (1 - size1 / size0) * 100
        print(f"  {p.name:32s} {size0/1024/1024:6.1f} MB -> {size1/1024/1024:5.1f} MB  ({ratio:+.0f}%)")
    print(f"\nTotal: {before/1024/1024:.1f} MB -> {after/1024/1024:.1f} MB "
          f"({(1 - after/before)*100:+.0f}%)")


if __name__ == "__main__":
    main()
