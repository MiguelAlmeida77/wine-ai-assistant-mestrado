"""Smoke + sanity tests for core models. Run with `pytest`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.loader import FEATURE_COLUMNS, load_raw
from src.data.preprocessor import add_engineered_features, basic_clean
from src.data.synthetic_aging import peak_year, project_chemistry, quality_curve
from src.models.blend_recommender import (
    BaseWine,
    BlendConstraints,
    mix_chemistry,
    recommend_blend,
)
from src.models.quality_predictor import MODELS_DIR, QualityModel


def _sample_chem() -> dict:
    return {
        "fixed acidity": 8.0,
        "volatile acidity": 0.4,
        "citric acid": 0.3,
        "residual sugar": 2.5,
        "chlorides": 0.07,
        "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 110.0,
        "density": 0.997,
        "pH": 3.3,
        "sulphates": 0.65,
        "alcohol": 12.5,
    }


def test_loader_dataset_sizes() -> None:
    df = load_raw("both")
    assert len(df) == 6497
    assert set(df["type"].unique()) == {"red", "white"}
    assert (df["type"] == "red").sum() == 1599
    assert (df["type"] == "white").sum() == 4898


def test_basic_clean_drops_duplicates() -> None:
    df = load_raw("red")
    cleaned = basic_clean(df)
    assert len(cleaned) <= len(df)
    assert cleaned.isna().sum().sum() == 0


def test_engineered_features_added() -> None:
    df = load_raw("red").head(100)
    eng = add_engineered_features(df)
    for f in ("total_acidity", "bound_sulfur_dioxide", "alcohol_sugar_ratio"):
        assert f in eng.columns


def test_mix_chemistry_linear() -> None:
    chem_a = _sample_chem()
    chem_b = dict(chem_a)
    chem_b["alcohol"] = 14.5
    a = BaseWine("a", chem_a)
    b = BaseWine("b", chem_b)
    mixed = mix_chemistry([a, b], np.array([0.5, 0.5]))
    assert mixed["alcohol"] == pytest.approx(13.5)


def test_project_chemistry_monotonic_so2() -> None:
    """Free SO2 must decrease monotonically with time."""
    chem = _sample_chem()
    so2 = [project_chemistry(chem, t, "red")["free sulfur dioxide"]
           for t in (0, 2, 5, 10)]
    assert all(so2[i] > so2[i + 1] for i in range(len(so2) - 1))


def test_project_chemistry_va_increases() -> None:
    """Volatile acidity must rise (or stay flat at cap) over time."""
    chem = _sample_chem()
    va = [project_chemistry(chem, t, "red")["volatile acidity"]
          for t in (0, 2, 5, 10)]
    assert all(va[i] <= va[i + 1] for i in range(len(va) - 1))


def test_peak_year_within_bounds() -> None:
    for wt in ("red", "white"):
        p = peak_year(_sample_chem(), wt)
        assert 1.0 <= p <= 20.0


def test_quality_curve_inverted_u() -> None:
    """The curve should have a single global maximum (inverted U)."""
    chem = _sample_chem()
    curve = quality_curve(chem, 6.0, "red", years_grid=np.arange(0, 15, 0.5))
    q = curve["predicted_quality"].values
    peak_idx = int(np.argmax(q))
    # Strictly non-decreasing before peak and non-increasing after (allow ties)
    assert all(q[i] <= q[i + 1] + 1e-9 for i in range(peak_idx))
    assert all(q[i] >= q[i + 1] - 1e-9 for i in range(peak_idx, len(q) - 1))


@pytest.mark.skipif(
    not (MODELS_DIR / "quality_red.joblib").exists(),
    reason="Trained model not present; run scripts/train_all.py first",
)
def test_quality_model_predicts_reasonably() -> None:
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    pred = float(model.predict(_sample_chem())[0])
    # Prediction should be within plausible UCI range (3–8 covers >99% of data)
    assert 3.0 <= pred <= 8.5


@pytest.mark.skipif(
    not (MODELS_DIR / "quality_red.joblib").exists(),
    reason="Trained model not present; run scripts/train_all.py first",
)
def test_blend_optimizer_respects_alcohol_constraint() -> None:
    model = QualityModel.load(MODELS_DIR / "quality_red.joblib")
    a = BaseWine("low", {**_sample_chem(), "alcohol": 10.5})
    b = BaseWine("high", {**_sample_chem(), "alcohol": 14.0})
    res = recommend_blend(
        [a, b], model,
        constraints=BlendConstraints(min_alcohol=12.5, max_alcohol=13.5),
    )
    assert res.converged
    assert 12.4 <= res.blended_chemistry["alcohol"] <= 13.6
    # Proportions sum to 1
    total = sum(res.proportions.values())
    assert total == pytest.approx(1.0, abs=1e-6)
