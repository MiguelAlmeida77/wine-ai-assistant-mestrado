"""Loader for the Wine Enthusiast reviews dataset (~130k samples).

Source: Kaggle "Wine Reviews" (zynicide), mirrored on tidytuesday.

Columns:
  country, description, designation, points (80–100), price (USD), province,
  region_1, region_2, taster_name, taster_twitter_handle, title, variety,
  winery.

Important caveats for the thesis:
  - `points` are from a single publication (Wine Enthusiast). They are a
    proxy for critic acceptance, not a sensory panel.
  - The 80–100 scale is left-censored (the magazine does not publish wines
    below 80) — a source of selection bias.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .loader import PROCESSED_DIR, RAW_DIR

REVIEWS_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
    "data/2019/2019-05-28/winemag-data-130k-v2.csv"
)
REVIEWS_LOCAL = RAW_DIR / "winemag-reviews.csv"


def download_reviews(force: bool = False) -> Path:
    REVIEWS_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if force:
        REVIEWS_LOCAL.unlink(missing_ok=True)
    if not REVIEWS_LOCAL.exists():
        r = requests.get(REVIEWS_URL, timeout=60)
        r.raise_for_status()
        REVIEWS_LOCAL.write_bytes(r.content)
    return REVIEWS_LOCAL


def load_reviews(min_complete: bool = True) -> pd.DataFrame:
    """Load and lightly clean the Wine Reviews dataset.

    `min_complete=True` (default) drops rows missing `price`, `country`,
    `variety` or `winery`. Those four are the minimum needed for the market
    model and dropping them removes ~10% of rows.
    """
    path = download_reviews()
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    if min_complete:
        df = df.dropna(subset=["price", "country", "variety", "winery", "points"])
    df = df.reset_index(drop=True)
    return df


def save_processed_reviews(df: pd.DataFrame, name: str = "reviews") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    df = load_reviews()
    print(f"Loaded {len(df)} reviews after cleaning")
    print(f"Countries: {df['country'].nunique()}, varieties: {df['variety'].nunique()}, "
          f"wineries: {df['winery'].nunique()}")
    out = save_processed_reviews(df)
    print(f"Saved to {out}")
