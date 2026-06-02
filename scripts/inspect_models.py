"""Imprime um resumo do estado de todos os artefactos do projeto.

Útil para verificar rapidamente o que está treinado, qual o tamanho dos
modelos e quais as métricas correntes — sem ter de abrir vários ficheiros.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quality_predictor import MODELS_DIR


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    print(f"== Diretoria: {MODELS_DIR} ==\n")
    if not MODELS_DIR.exists():
        print("Sem modelos. Corra `scripts\\train_all.py`.")
        return

    artefacts = sorted(MODELS_DIR.iterdir())
    print(f"{'Ficheiro':<40} {'Tamanho':>10}  Última modificação")
    print("-" * 80)
    for p in artefacts:
        size = _human(p.stat().st_size)
        mtime = p.stat().st_mtime
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{p.name:<40} {size:>10}  {ts}")

    print("\n== Métricas dos modelos ==\n")

    def _show(name: str, path: Path):
        if not path.exists():
            print(f"[{name}] -- não treinado --")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{name}]")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print()

    _show("quality", MODELS_DIR / "quality_metrics.json")
    _show("intervalos (incerteza)", MODELS_DIR / "interval_metrics.json")
    _show("envelhecimento", MODELS_DIR / "aging_metrics.json")
    _show("mercado", MODELS_DIR / "market_metrics.json")
    _show("Optuna — resumo", MODELS_DIR / "best_params_summary.json")


if __name__ == "__main__":
    main()
