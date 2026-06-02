"""Download and load UCI Wine Quality datasets (red + white)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import requests

# GitHub mirrors used because UCI's HTTPS endpoint has TLS handshake issues
# in some Windows environments. Both files are the original Cortez et al. CSVs.
RED_URL = "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv"
WHITE_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/winequality-white.csv"
)

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Portuguese display labels for UI
FEATURE_LABELS_PT = {
    "fixed acidity": "Acidez fixa (g/L)",
    "volatile acidity": "Acidez volátil (g/L)",
    "citric acid": "Ácido cítrico (g/L)",
    "residual sugar": "Açúcar residual (g/L)",
    "chlorides": "Cloretos (g/L)",
    "free sulfur dioxide": "SO₂ livre (mg/L)",
    "total sulfur dioxide": "SO₂ total (mg/L)",
    "density": "Densidade (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfatos (g/L)",
    "alcohol": "Álcool (% vol.)",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download_uci(force: bool = False) -> tuple[Path, Path]:
    red_path = RAW_DIR / "winequality-red.csv"
    white_path = RAW_DIR / "winequality-white.csv"
    if force:
        red_path.unlink(missing_ok=True)
        white_path.unlink(missing_ok=True)
    _download(RED_URL, red_path)
    _download(WHITE_URL, white_path)
    return red_path, white_path


def load_raw(kind: Literal["red", "white", "both"] = "both") -> pd.DataFrame:
    red_path, white_path = download_uci()
    frames: list[pd.DataFrame] = []
    if kind in ("red", "both"):
        red = pd.read_csv(red_path, sep=None, engine="python", encoding="utf-8-sig")
        red["type"] = "red"
        frames.append(red)
    if kind in ("white", "both"):
        white = pd.read_csv(white_path, sep=None, engine="python", encoding="utf-8-sig")
        white["type"] = "white"
        frames.append(white)
    df = pd.concat(frames, ignore_index=True)
    return df


def save_processed(df: pd.DataFrame, name: str = "wines") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    df = load_raw("both")
    out = save_processed(df)
    print(f"Loaded {len(df)} samples ({df['type'].value_counts().to_dict()})")
    print(f"Saved to {out}")
