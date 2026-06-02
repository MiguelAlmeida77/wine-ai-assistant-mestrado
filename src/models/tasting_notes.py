"""Gerador de notas de prova a partir do dataset Wine Reviews.

Abordagem (defensável academicamente):

1. **Léxico curado** de ~60 descritores agrupados em famílias (frutos, flores,
   especiarias, mineral, madeira, etc.). Cada descritor mapeia uma lista de
   *keywords* em inglês (linguagem dos reviews).

2. **Extração multi-label** do campo `description` do Wine Reviews: para cada
   review, marcamos 1 se qualquer keyword aparece no texto.

3. **Modelo**: regressor multi-saída (XGBoost por descritor, ou um único
   MultiOutputRegressor) que mapeia *(país, casta, província, log preço)*
   para a probabilidade de cada descritor.

4. **Geração**: dado um vinho com química UCI, o utilizador escolhe metadata
   (casta, país, preço). O modelo devolve probabilidades por descritor; a
   nota é montada por templates simples em português.

Limitações documentadas:
  - Não usa diretamente a química (Wine Reviews não tem); a ponte é o
    metadata escolhido pelo utilizador.
  - Léxico é manualmente curado e limitado a 60 descritores frequentes.
  - Probabilidades são marginais (descritores assumidos independentes).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..data.reviews_loader import load_reviews
from .market_predictor import CAT_COLS, NUM_COLS, _cap_categories
from .quality_predictor import MODELS_DIR

# --- Léxico de descritores enológicos ---------------------------------------
# Cada chave é um descritor canónico em português; o valor é a lista de
# *keywords* (regex word-boundary) a procurar no texto inglês do review.
DESCRIPTOR_LEXICON: dict[str, list[str]] = {
    # Frutos
    "frutos vermelhos": ["red fruit", "cherry", "strawberry", "raspberry", "currant"],
    "frutos negros": ["black fruit", "blackberry", "plum", "blueberry", "cassis"],
    "frutos tropicais": ["tropical", "pineapple", "mango", "passion fruit", "guava"],
    "frutos cítricos": ["citrus", "lemon", "lime", "grapefruit", "orange peel"],
    "maçã": ["apple"],
    "pera": ["pear"],
    "pêssego": ["peach", "nectarine"],
    # Flores
    "floral": ["floral", "flower", "blossom"],
    "rosa": ["rose"],
    "violeta": ["violet"],
    # Especiarias
    "pimenta": ["pepper", "peppery"],
    "canela": ["cinnamon"],
    "baunilha": ["vanilla"],
    "cravinho": ["clove"],
    "noz-moscada": ["nutmeg"],
    # Madeira / estágio
    "madeira": ["oak", "wood", "barrel"],
    "tostado": ["toast", "toasted", "smoke", "smoky"],
    "torrefação": ["coffee", "espresso", "chocolate", "cocoa"],
    # Mineralidade
    "mineral": ["mineral", "flint", "stone", "wet stones"],
    "salgado": ["saline", "salty"],
    # Vegetal / herbáceo
    "herbáceo": ["herbal", "herbs", "thyme", "sage", "rosemary"],
    "vegetal": ["green", "grassy", "vegetal", "bell pepper"],
    "tabaco": ["tobacco"],
    "couro": ["leather"],
    "terroso": ["earthy", "earth", "forest floor", "mushroom"],
    # Estrutura
    "taninos firmes": ["firm tannins", "grippy", "tannic", "tannins"],
    "taninos suaves": ["soft tannins", "silky", "smooth"],
    "acidez elevada": ["bright acidity", "crisp", "zesty", "vibrant acidity"],
    "acidez equilibrada": ["balanced acidity", "well balanced"],
    "corpo encorpado": ["full bodied", "full-bodied", "rich", "concentrated"],
    "corpo leve": ["light bodied", "light-bodied", "delicate"],
    "final longo": ["long finish", "lingering", "persistent finish"],
    "final mineral": ["mineral finish", "stony finish"],
    "final amargo": ["bitter finish", "bitterness"],
    # Doce / seco
    "seco": ["dry"],
    "frutado": ["fruity", "fruit forward"],
    "doce": ["sweet", "sweetness"],
    "fresco": ["fresh", "refreshing"],
}

# Famílias para apresentação na UI
DESCRIPTOR_FAMILIES: dict[str, list[str]] = {
    "Frutos": ["frutos vermelhos", "frutos negros", "frutos tropicais", "frutos cítricos",
               "maçã", "pera", "pêssego"],
    "Flores": ["floral", "rosa", "violeta"],
    "Especiarias": ["pimenta", "canela", "baunilha", "cravinho", "noz-moscada"],
    "Estágio": ["madeira", "tostado", "torrefação"],
    "Mineralidade": ["mineral", "salgado"],
    "Vegetal": ["herbáceo", "vegetal", "tabaco", "couro", "terroso"],
    "Estrutura": ["taninos firmes", "taninos suaves", "acidez elevada",
                  "acidez equilibrada", "corpo encorpado", "corpo leve",
                  "final longo", "final mineral", "final amargo"],
    "Estilo": ["seco", "frutado", "doce", "fresco"],
}

DESCRIPTORS = list(DESCRIPTOR_LEXICON.keys())


def extract_descriptors(text: str) -> np.ndarray:
    """Return 0/1 vector aligned with DESCRIPTORS for a single review."""
    t = (text or "").lower()
    out = np.zeros(len(DESCRIPTORS), dtype=np.int8)
    for i, d in enumerate(DESCRIPTORS):
        for kw in DESCRIPTOR_LEXICON[d]:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                out[i] = 1
                break
    return out


@dataclass
class NotesModel:
    pipeline: Pipeline
    descriptors: list[str] = field(default_factory=lambda: list(DESCRIPTORS))
    metrics: dict = field(default_factory=dict)

    def predict_probs(self, sample: dict | pd.DataFrame) -> pd.Series:
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        for c in CAT_COLS:
            if c not in sample.columns:
                sample[c] = "unknown"
        if "price" in sample.columns and "log_price" not in sample.columns:
            sample = sample.copy()
            sample["log_price"] = np.log1p(sample["price"].fillna(20.0))
        if "log_price" not in sample.columns:
            sample = sample.copy()
            sample["log_price"] = np.log1p(20.0)
        # MultiOutputClassifier.predict_proba returns list of (n_samples, 2)
        probas = self.pipeline.predict_proba(sample)
        out = np.array([p[:, 1][0] if p.shape[1] == 2 else 0.0 for p in probas])
        return pd.Series(out, index=self.descriptors)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "NotesModel":
        return joblib.load(path)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_price"] = np.log1p(df["price"].fillna(df["price"].median()))
    for c, n in {"country": 25, "variety": 40, "province": 50, "taster_name": 20}.items():
        out[c] = _cap_categories(df, c, n)
    return out


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=400, C=1.0, n_jobs=-1, solver="lbfgs"),
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train(seed: int = 42, sample: int | None = 40_000) -> NotesModel:
    df = load_reviews()
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    print(f"Extracting descriptors from {len(df)} reviews...")
    Y = np.stack([extract_descriptors(t) for t in df["description"]])
    descriptor_freq = Y.mean(axis=0)
    # Drop ultra-rare descriptors that yield degenerate single-class targets
    valid = descriptor_freq >= 0.01
    Y = Y[:, valid]
    descriptors = [d for d, ok in zip(DESCRIPTORS, valid) if ok]
    print(f"  kept {len(descriptors)} descriptors with frequency ≥ 1%")

    X = _prepare(df)
    groups = df["winery"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(splitter.split(X, Y, groups=groups))

    pipe = _build_pipeline()
    pipe.fit(X.iloc[tr], Y[tr])
    pred = pipe.predict(X.iloc[te])
    f1_micro = float(f1_score(Y[te], pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(Y[te], pred, average="macro", zero_division=0))
    metrics = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_descriptors": int(len(descriptors)),
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "frequencies": {d: float(descriptor_freq[i]) for i, d in enumerate(DESCRIPTORS) if valid[i]},
    }
    model = NotesModel(pipeline=pipe, descriptors=descriptors, metrics=metrics)
    model.save(MODELS_DIR / "notes.joblib")
    (MODELS_DIR / "notes_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model


def render_note(
    probs: pd.Series,
    threshold: float = 0.12,
    top_k: int = 14,
    base_frequencies: dict | None = None,
    lift_threshold: float = 1.4,
) -> str:
    """Compose a Portuguese tasting note from descriptor probabilities.

    A descriptor is included if EITHER (a) its absolute probability >= threshold
    OR (b) when `base_frequencies` is supplied, its lift over base
    (prob / base_freq) >= lift_threshold. This brings in rare-but-distinctive
    descriptors that absolute thresholds would miss.
    """
    candidates = probs.copy()
    if base_frequencies:
        base = pd.Series(base_frequencies).reindex(candidates.index).fillna(0.5)
        lift = candidates / base.clip(lower=0.01)
        keep = (candidates >= threshold) | (lift >= lift_threshold)
        candidates = candidates[keep]
    else:
        candidates = candidates[candidates >= threshold]
    s = candidates.sort_values(ascending=False).head(top_k)
    if s.empty:
        return "Sem descritores dominantes — vinho de perfil pouco marcado."

    by_family: dict[str, list[str]] = {}
    for fam, members in DESCRIPTOR_FAMILIES.items():
        for d in members:
            if d in s.index:
                by_family.setdefault(fam, []).append(d)

    parts: list[str] = []
    if any(f in by_family for f in ("Frutos", "Flores", "Especiarias")):
        aromas = []
        for fam in ("Frutos", "Flores", "Especiarias"):
            aromas.extend(by_family.get(fam, []))
        if aromas:
            parts.append("Notas aromáticas de " + ", ".join(aromas) + ".")
    if "Estágio" in by_family:
        parts.append("Estágio em madeira evidente: " + ", ".join(by_family["Estágio"]) + ".")
    if "Mineralidade" in by_family or "Vegetal" in by_family:
        terr = by_family.get("Mineralidade", []) + by_family.get("Vegetal", [])
        if terr:
            parts.append("Acentos de " + ", ".join(terr) + ".")
    if "Estrutura" in by_family:
        parts.append("Em boca: " + ", ".join(by_family["Estrutura"]) + ".")
    if "Estilo" in by_family:
        parts.append("Estilo: " + ", ".join(by_family["Estilo"]) + ".")
    return " ".join(parts)


if __name__ == "__main__":
    model = train()
    print(json.dumps(model.metrics, indent=2, ensure_ascii=False))
    sample = {
        "country": "Portugal", "variety": "Touriga Nacional",
        "province": "Douro", "taster_name": "Roger Voss", "price": 35.0,
    }
    probs = model.predict_probs(sample)
    print("\nTop descritores:")
    print(probs.sort_values(ascending=False).head(10).round(3))
    print("\nNota gerada:")
    print(render_note(probs))
