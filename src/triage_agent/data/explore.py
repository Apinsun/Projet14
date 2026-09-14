"""Fonctions d'analyse exploratoire (EDA) des datasets bruts.

Le but est de produire, pour chaque corpus, un résumé factuel :

- nombre de lignes par split et schéma (colonnes + types) ;
- valeurs manquantes ;
- statistiques de longueur des champs textuels ;
- détection de langue (FR/EN) sur les champs textuels clés ;
- détection de doublons (au niveau question / prompt) ;
- distributions propres à chaque corpus (types de questions, sources, etc.).

Les résultats sont agrégés dans un dictionnaire sérialisable en JSON, consommé
par ``scripts/run_eda.py`` pour générer le rapport Markdown.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from triage_agent.data.config import DATASETS, RAW_DIR

# Seed déterministe pour langdetect.
DetectorFactory.seed = 0

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def load_raw(name: str):
    """Charge les fichiers Parquet bruts d'un dataset en ``DatasetDict``.

    Le nom de chaque split est dérivé du nom de fichier (``train.parquet`` ->
    ``train``).
    """
    from datasets import Dataset, DatasetDict

    splits = {}
    for p in sorted((RAW_DIR / name).glob("*.parquet")):
        splits[p.stem] = Dataset.from_parquet(str(p))
    return DatasetDict(splits)


def _detect_lang(text: str | None) -> str:
    if not text or not text.strip():
        return "empty"
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


def _len_stats(values: pd.Series) -> dict[str, float]:
    """Statistiques de longueur (en caractères) d'une série de textes."""
    lengths = values.astype(str).str.len()
    return {
        "count": int(lengths.count()),
        "mean": round(float(lengths.mean()), 1),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }


def _pct_series(series: pd.Series, top: int = 20) -> dict[str, Any]:
    counts = series.value_counts(dropna=False).head(top)
    total = len(series)
    return {str(k): {"count": int(v), "pct": round(100.0 * v / total, 2)} for k, v in counts.items()}


def _html_count(series: pd.Series) -> int:
    return int(series.astype(str).str.contains(_HTML_RE, regex=True).sum())


def summarize(name: str, sample_n: int = 400) -> dict[str, Any]:
    """Produit le résumé EDA d'un dataset."""
    spec = DATASETS[name]
    ds = load_raw(name)
    # Split représentatif pour l'analyse : "train" sinon le plus gros split.
    rep_split = "train" if "train" in ds else max(ds, key=lambda s: len(ds[s]))
    df = ds[rep_split].to_pandas()

    info: dict[str, Any] = {
        "name": name,
        "hf_id": spec["hf_id"],
        "language": spec["language"],
        "role": spec["role"],
        "splits": {split: len(subset) for split, subset in ds.items()},
        "columns": {col: str(df[col].dtype) for col in df.columns},
        "num_rows_representative_split": len(df),
        "representative_split": rep_split,
    }

    # Valeurs manquantes (sur le split train).
    info["missing"] = {col: int(df[col].isna().sum()) for col in df.columns}

    # Doublons sur la colonne "question" ou "prompt" selon disponibilité.
    dup_col = "question" if "question" in df.columns else ("prompt" if "prompt" in df.columns else None)
    if dup_col:
        info["duplicates"] = {
            "total_rows": len(df),
            "unique_values": int(df[dup_col].nunique()),
            "duplicated_rows": int(df[dup_col].duplicated().sum()),
        }

    # Langue détectée sur un échantillon.
    lang_col = "question" if "question" in df.columns else ("prompt" if "prompt" in df.columns else None)
    if lang_col:
        sample = df[lang_col].dropna().head(sample_n)
        langs = sample.apply(_detect_lang)
        info["language_distribution"] = _pct_series(langs)

    # Longueurs des champs textuels (colonnes de type chaîne uniquement).
    info["lengths"] = {}
    for col in df.columns:
        first = df[col].dropna().iloc[0] if df[col].notna().any() else None
        if isinstance(first, str):
            info["lengths"][col] = _len_stats(df[col])

    # Balises HTML résiduelles.
    info["html_tags"] = {
        col: _html_count(df[col])
        for col in df.columns
        if df[col].notna().any() and isinstance(df[col].dropna().iloc[0], str)
    }

    # Analyses spécifiques par corpus.
    info["specific"] = _specific_analysis(name, df)

    # Échantillons (tronqués) pour le rapport.
    info["samples"] = _samples(name, df)

    return info


def _specific_analysis(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Distributions spécifiques à chaque corpus."""
    out: dict[str, Any] = {}
    if name == "frenchmedmcqa":
        out["nbr_correct_answers"] = _pct_series(df["nbr_correct_answers"].astype(str))
        out["type"] = _pct_series(df["type"].astype(str))
        out["subject_name"] = _pct_series(df["subject_name"].astype(str))
    elif name == "medquad":
        out["question_type"] = _pct_series(df["question_type"])
        out["document_source"] = _pct_series(df["document_source"])
        out["question_focus_top"] = _pct_series(df["question_focus"], top=10)
    elif name == "mediqa":
        out["type"] = _pct_series(df["type"])
        out["n_answers"] = _pct_series(
            df["answer"].apply(lambda a: len(list(a)) if a is not None else 0).astype(str)
        )
        out["empty_context"] = int(
            df["context"].isna().sum() + (df["context"].astype(str).str.strip() == "").sum()
        )
        out["empty_choices"] = int(
            df["choices"].apply(lambda c: len(list(c)) if c is not None else 0).sum()
        )
        # Longueur cumulée des réponses (champ de type liste).
        out["answer_len"] = _len_stats(
            df["answer"].apply(lambda a: sum(len(x) for x in list(a)) if a is not None else 0).astype(str)
        )
    elif name == "ultramedical_preference":
        def _roles(x):
            lst = list(x) if x is not None else []
            return lst[-1]["role"] if lst else None

        def _content_text(x):
            lst = list(x) if x is not None else []
            return lst[-1]["content"] if lst else ""

        out["label_type"] = _pct_series(df["label_type"])
        out["chosen_role"] = _pct_series(df["chosen"].apply(_roles))
        out["rejected_role"] = _pct_series(df["rejected"].apply(_roles))
        out["chosen_model"] = _pct_series(
            df["metadata"].apply(lambda m: m["chosen"]["model"] if m and "chosen" in m else None),
            top=15,
        )
        out["rejected_model"] = _pct_series(
            df["metadata"].apply(lambda m: m["rejected"]["model"] if m and "rejected" in m else None),
            top=15,
        )
        out["chosen_content_len"] = _len_stats(df["chosen"].apply(_content_text).astype(str))
        out["rejected_content_len"] = _len_stats(df["rejected"].apply(_content_text).astype(str))
        out["feedback_mean_len"] = _len_stats(df["feedback"].astype(str))
    return out


def _samples(name: str, df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    """Échantillons représentatifs tronqués à 300 caractères."""
    limit = 300
    out = []
    for _, row in df.head(n).iterrows():
        d = {}
        for col in df.columns:
            v = row[col]
            s = str(v)
            d[col] = s[:limit] + ("…" if len(s) > limit else "")
        out.append(d)
    return out


def summarize_all() -> dict[str, Any]:
    """Résumé EDA de tous les datasets."""
    return {name: summarize(name) for name in DATASETS}
