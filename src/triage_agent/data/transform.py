"""Standardisation et nettoyage des datasets bruts vers le schéma JSONL cible.

Chaque source produit un fichier JSONL dans ``data/processed/`` :

- ``frenchmedmcqa.jsonl``             (SFT, FR) — QCM -> paire instruction/réponse ;
- ``medquad.jsonl``                   (SFT, EN) — question -> réponse (filtre réponses nulles) ;
- ``mediqa.jsonl``                    (SFT, EN) — question -> réponse « gold » (ReferenceRank=1) ;
- ``ultramedical_preference.jsonl``   (DPO, EN) — {prompt, chosen, rejected}.

Schéma SFT (voir rapport §5) :

.. code-block:: json

    {
      "id": "source::id_original",
      "lang": "fr|en",
      "source": "...",
      "role": "sft",
      "messages": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}],
      "metadata": {"symptoms": [], "medical_history": [], "vital_signs": {},
                   "topic": null, "question_type": "...", "source_id": "...",
                   "split": "...", "license": "...", "confidence": null}
    }

Schéma DPO :

.. code-block:: json

    {"id": "...", "lang": "en", "source": "ultramedical_preference", "role": "dpo",
     "prompt": "...", "chosen": "...", "rejected": "...",
     "metadata": {"label_type": "...", "chosen_model": "...", "rejected_model": "...",
                  "source_id": "...", "split": "...", "license": "mit", "confidence": null}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from triage_agent.data.clean import clean_text
from triage_agent.data.config import DATASETS, PROCESSED_DIR, RAW_DIR
from triage_agent.data.download import download_parquet_config

logger = logging.getLogger("triage_agent.data.transform")

# Seuils de nettoyage (en caractères).
MIN_TEXT = 3
MAX_QUESTION = 2_000
MAX_ANSWER = 4_000
MAX_DPO_PROMPT = 2_000

# Lettres des options FrenchMedMCQA -> colonnes du parquet.
_OPT_KEYS = {"a": "answer_a", "b": "answer_b", "c": "answer_c", "d": "answer_d", "e": "answer_e"}

# Config MediQA (source) conservant le classement humain ReferenceRank.
_MEDIQA_SOURCE_CONFIG = "mediqa_qa_source"
_MEDIQA_SOURCE_SPLITS = ["train_live_qa_med", "train_alexa", "validation", "test"]


def _load_pandas(name: str, split: str) -> pd.DataFrame:
    """Charge un split Parquet d'un dataset en DataFrame pandas."""
    return pd.read_parquet(RAW_DIR / name / f"{split}.parquet")


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sft_record(lang: str, source: str, messages: list[dict], meta: dict) -> dict:
    """Construit un enregistrement SFT au schéma cible."""
    return {
        "id": f"{source}::{meta['source_id']}",
        "lang": lang,
        "source": source,
        "role": "sft",
        "messages": messages,
        "metadata": {
            "symptoms": [],
            "medical_history": [],
            "vital_signs": {},
            "topic": meta.get("topic"),
            "question_type": meta.get("question_type"),
            "source_id": meta["source_id"],
            "split": meta.get("split"),
            "license": meta.get("license"),
            "confidence": meta.get("confidence"),
        },
    }


def _dpo_record(prompt: str, chosen: str, rejected: str, meta: dict) -> dict:
    """Construit un enregistrement DPO au schéma cible."""
    return {
        "id": f"ultramedical_preference::{meta['source_id']}",
        "lang": "en",
        "source": "ultramedical_preference",
        "role": "dpo",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "label_type": meta.get("label_type"),
            "chosen_model": meta.get("chosen_model"),
            "rejected_model": meta.get("rejected_model"),
            "source_id": meta["source_id"],
            "split": meta.get("split"),
            "license": "mit",
            "confidence": None,
        },
    }


# --------------------------------------------------------------------------- #
# FrenchMedMCQA
# --------------------------------------------------------------------------- #
def transform_frenchmedmcqa() -> tuple[list[dict], dict[str, int]]:
    """QCM -> paire instruction/réponse (concaténation des bonnes options)."""
    records: list[dict] = []
    stats = {"input": 0, "output": 0, "dropped": 0}

    for split in ["train", "validation", "test"]:
        df = _load_pandas("frenchmedmcqa", split)
        for _, row in df.iterrows():
            stats["input"] += 1
            question = clean_text(row["question"])
            options = {letter: clean_text(row[_OPT_KEYS[letter]]) for letter in "abcde"}
            correct = [c for c in row["correct_answers"] if c in options]

            if len(question) < MIN_TEXT or not correct:
                stats["dropped"] += 1
                continue

            option_lines = "\n".join(f"({letter.upper()}) {options[letter]}" for letter in "abcde")
            user = f"{question}\n{option_lines}"
            if len(correct) == 1:
                answer = f"La réponse correcte est : {options[correct[0]]}."
            else:
                joined = ", ".join(options[c] for c in correct)
                answer = f"Les réponses correctes sont : {joined}."

            records.append(
                _sft_record(
                    "fr",
                    "frenchmedmcqa",
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                    {
                        "topic": row["subject_name"] or "pharmacie",
                        "question_type": "mcq",
                        "source_id": row["id"],
                        "split": split,
                        "license": "apache-2.0",
                        "confidence": None,
                    },
                )
            )
            stats["output"] += 1
    return records, stats


# --------------------------------------------------------------------------- #
# MedQuAD
# --------------------------------------------------------------------------- #
def transform_medquad() -> tuple[list[dict], dict[str, int]]:
    """Question -> réponse ; filtre les réponses nulles, déduplique, nettoie."""
    records: list[dict] = []
    stats = {"input": 0, "output": 0, "dropped": 0}
    seen: set[str] = set()

    df = _load_pandas("medquad", "train")
    df = df[df["answer"].notna()]
    stats["input"] = len(df)

    for _, row in df.iterrows():
        question = clean_text(row["question"])
        answer = clean_text(row["answer"], strip_citations=True)

        if not (MIN_TEXT <= len(question) <= MAX_QUESTION and MIN_TEXT <= len(answer) <= MAX_ANSWER):
            stats["dropped"] += 1
            continue
        if question in seen:
            stats["dropped"] += 1
            continue
        seen.add(question)

        records.append(
            _sft_record(
                "en",
                "medquad",
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                {
                    "topic": row["question_focus"],
                    "question_type": row["question_type"],
                    "source_id": row["question_id"],
                    "split": "train",
                    "license": None,
                    "confidence": None,
                },
            )
        )
        stats["output"] += 1
    return records, stats


# --------------------------------------------------------------------------- #
# MediQA
# --------------------------------------------------------------------------- #
def _ensure_mediqa_source() -> Path:
    """Télécharge le config ``source`` de MediQA (ReferenceRank) si absent."""
    dest = RAW_DIR / "mediqa" / "source"
    if not any(dest.glob("*.parquet")):
        logger.info("Téléchargement du config source de MediQA (%s)", _MEDIQA_SOURCE_CONFIG)
        download_parquet_config(
            DATASETS["mediqa"]["hf_id"], _MEDIQA_SOURCE_CONFIG, dest,
        )
    return dest


def _mediqa_gold_answer(answer_list: Any) -> dict | None:
    """Retourne la réponse « gold » (ReferenceRank minimal, puis ReferenceScore max)."""
    best: dict | None = None
    for item in list(answer_list) if answer_list is not None else []:
        ans = item.get("Answer") if isinstance(item, dict) else None
        if not isinstance(ans, dict) or not ans.get("AnswerText"):
            continue
        if best is None:
            best = ans
            continue
        rank = ans.get("ReferenceRank") or 99
        score = ans.get("ReferenceScore") or 0
        best_rank = best.get("ReferenceRank") or 99
        best_score = best.get("ReferenceScore") or 0
        if rank < best_rank or (rank == best_rank and score > best_score):
            best = ans
    return best


def transform_mediqa() -> tuple[list[dict], dict[str, int]]:
    """Question -> réponse gold (ReferenceRank=1) à partir du config source."""
    records: list[dict] = []
    stats = {"input": 0, "output": 0, "dropped": 0}
    src_dir = _ensure_mediqa_source()

    for split in _MEDIQA_SOURCE_SPLITS:
        df = pd.read_parquet(src_dir / f"{split}.parquet")
        for _, row in df.iterrows():
            stats["input"] += 1
            q = row["QUESTION"]
            question = clean_text(q["QuestionText"])
            gold = _mediqa_gold_answer(q.get("AnswerList"))
            if gold is None:
                stats["dropped"] += 1
                continue
            answer = clean_text(gold["AnswerText"], strip_citations=True)

            if not (MIN_TEXT <= len(question) <= MAX_QUESTION and MIN_TEXT <= len(answer) <= MAX_ANSWER):
                stats["dropped"] += 1
                continue

            records.append(
                _sft_record(
                    "en",
                    "mediqa",
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    {
                        "topic": None,
                        "question_type": "factoid",
                        "source_id": q.get("QID"),
                        "split": split,
                        "license": None,
                        "confidence": gold.get("ReferenceScore"),
                    },
                )
            )
            stats["output"] += 1
    return records, stats


# --------------------------------------------------------------------------- #
# UltraMedical-Preference (DPO)
# --------------------------------------------------------------------------- #
def _assistant_content(messages: Any) -> str:
    """Concatène les contenus des messages de rôle ``assistant``."""
    lst = list(messages) if messages is not None else []
    parts = [m["content"] for m in lst if isinstance(m, dict) and m.get("role") == "assistant"]
    if parts:
        return "\n".join(parts)
    return lst[-1]["content"] if lst else ""


def _model(metadata: Any, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    sub = metadata.get(key)
    return sub.get("model") if isinstance(sub, dict) else None


def transform_ultramedical() -> tuple[list[dict], dict[str, int]]:
    """Nettoie, déduplique (sur prompt) et standardise les paires DPO."""
    records: list[dict] = []
    stats = {"input": 0, "output": 0, "dropped": 0}
    seen: set[str] = set()

    for split in ["train", "validation", "test"]:
        df = _load_pandas("ultramedical_preference", split)
        for _, row in df.iterrows():
            stats["input"] += 1
            prompt = clean_text(row["prompt"])
            chosen = clean_text(_assistant_content(row["chosen"]), strip_citations=True)
            rejected = clean_text(_assistant_content(row["rejected"]), strip_citations=True)

            if not (
                MIN_TEXT <= len(prompt) <= MAX_DPO_PROMPT
                and MIN_TEXT <= len(chosen) <= MAX_ANSWER
                and MIN_TEXT <= len(rejected) <= MAX_ANSWER
            ):
                stats["dropped"] += 1
                continue
            if prompt in seen:
                stats["dropped"] += 1
                continue
            seen.add(prompt)

            records.append(
                _dpo_record(
                    prompt,
                    chosen,
                    rejected,
                    {
                        "label_type": row["label_type"],
                        "chosen_model": _model(row["metadata"], "chosen"),
                        "rejected_model": _model(row["metadata"], "rejected"),
                        "source_id": row["prompt_id"],
                        "split": split,
                    },
                )
            )
            stats["output"] += 1
    return records, stats


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_TRANSFORMERS = {
    "frenchmedmcqa": transform_frenchmedmcqa,
    "medquad": transform_medquad,
    "mediqa": transform_mediqa,
    "ultramedical_preference": transform_ultramedical,
}


def transform_all() -> dict[str, Any]:
    """Exécute toutes les transformations et écrit les JSONL + le manifeste."""
    manifest: dict[str, Any] = {"generated_files": {}, "stats": {}}

    for name, fn in _TRANSFORMERS.items():
        logger.info("Transformation de %s ...", name)
        records, stats = fn()
        out = PROCESSED_DIR / f"{name}.jsonl"
        _write_jsonl(records, out)
        manifest["generated_files"][name] = str(out)
        manifest["stats"][name] = stats
        logger.info(
            "  %s -> %s (%d lignes ; %d écartées)",
            name,
            out,
            stats["output"],
            stats["dropped"],
        )

    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifeste écrit dans %s", manifest_path)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transform_all()
