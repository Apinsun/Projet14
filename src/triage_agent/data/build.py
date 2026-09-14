"""Constitution des jeux finaux SFT (~5 000), DPO et d'évaluation clinique.

Depuis les fichiers JSONL standardisés (``data/processed/*.jsonl``), on sélectionne :

- **SFT** : ~2 400 paires FR (FrenchMedMCQA train+validation) + ~2 600 EN
  (MedQuAD train + MediQA train/validation), en **excluant les splits ``test``**
  des sources (réservés à l'évaluation clinique).
- **DPO** : un sous-ensemble de paires UltraMedical-Preference (train+validation),
  échantillonné de façon stratifiée sur ``label_type``.
- **Évaluation clinique isolée** (jamais vue en entraînement) : FrenchMedMCQA
  ``test`` + MediQA ``test`` + un échantillon réservé de MedQuAD.

Le découpage SFT train/validation (90/10) est stratifié sur la langue et la
source pour préserver le ratio FR/EN.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from triage_agent.data.config import PROCESSED_DIR

logger = logging.getLogger("triage_agent.data.build")

FR_N = 2_400
EN_N = 2_600
DPO_N = 5_000
MEDQUAD_EVAL_RESERVE = 300
SEED = 42

# Splits utilisables pour l'entraînement SFT (hors test).
_FR_TRAIN_SPLITS = {"train", "validation"}
_MEDIQA_TRAIN_SPLITS = {"train_live_qa_med", "train_alexa", "validation"}
_DPO_TRAIN_SPLITS = {"train", "validation"}

SELECTED_DIR = PROCESSED_DIR / "selected"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _stratified_sample(records: list[dict], n: int, key: str, rng: random.Random) -> list[dict]:
    """Échantillonne ``n`` enregistrements en stratifiant sur ``key`` (ex. label_type)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r["metadata"].get(key))].append(r)

    selected: list[dict] = []
    remaining = n
    group_names = list(groups)
    for g in group_names[:-1]:
        # Part proportionnelle à la taille du groupe.
        take = round(n * len(groups[g]) / len(records))
        take = min(take, len(groups[g]), remaining)
        selected.extend(rng.sample(groups[g], take))
        remaining -= take
    # Dernier groupe (ou restant) : combler.
    last = group_names[-1] if group_names else None
    if remaining > 0 and last is not None:
        selected.extend(rng.sample(groups[last], min(remaining, len(groups[last]))))
    return selected


def select_sft() -> tuple[list[dict], dict[str, Any]]:
    """Sélectionne les paires SFT FR + EN et réserve l'échantillon MedQuAD d'éval."""
    rng = random.Random(SEED)

    fr_all = [
        r for r in load_jsonl(PROCESSED_DIR / "frenchmedmcqa.jsonl")
        if r["metadata"]["split"] in _FR_TRAIN_SPLITS
    ]
    medquad_all = load_jsonl(PROCESSED_DIR / "medquad.jsonl")
    mediqa_all = [
        r for r in load_jsonl(PROCESSED_DIR / "mediqa.jsonl")
        if r["metadata"]["split"] in _MEDIQA_TRAIN_SPLITS
    ]

    # Réserve d'évaluation clinique MedQuAD (hors pool d'entraînement).
    medquad_shuffled = medquad_all[:]
    rng.shuffle(medquad_shuffled)
    medquad_eval = medquad_shuffled[:MEDQUAD_EVAL_RESERVE]
    medquad_train = medquad_shuffled[MEDQUAD_EVAL_RESERVE:]

    fr_sel = rng.sample(fr_all, min(FR_N, len(fr_all)))
    en_pool = medquad_train + mediqa_all
    en_sel = rng.sample(en_pool, min(EN_N, len(en_pool)))

    sft = fr_sel + en_sel
    rng.shuffle(sft)

    stats = {
        "fr": {"available": len(fr_all), "selected": len(fr_sel)},
        "en": {"available": len(en_pool), "selected": len(en_sel)},
        "medquad_eval_reserve": len(medquad_eval),
        "total": len(sft),
    }
    return sft, stats, medquad_eval  # type: ignore[return-value]


def select_dpo() -> tuple[list[dict], dict[str, Any]]:
    """Sélectionne un sous-ensemble DPO stratifié sur ``label_type``."""
    rng = random.Random(SEED)
    pool = [
        r for r in load_jsonl(PROCESSED_DIR / "ultramedical_preference.jsonl")
        if r["metadata"]["split"] in _DPO_TRAIN_SPLITS
    ]
    selected = _stratified_sample(pool, DPO_N, "label_type", rng)
    rng.shuffle(selected)

    label_dist = Counter(r["metadata"]["label_type"] for r in selected)
    stats = {"available": len(pool), "selected": len(selected), "label_type": dict(label_dist)}
    return selected, stats


def select_clinical_eval(medquad_eval: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Constitue le jeu d'évaluation clinique isolé (splits test + réserve MedQuAD)."""
    fr_test = [r for r in load_jsonl(PROCESSED_DIR / "frenchmedmcqa.jsonl") if r["metadata"]["split"] == "test"]
    mediqa_test = [r for r in load_jsonl(PROCESSED_DIR / "mediqa.jsonl") if r["metadata"]["split"] == "test"]

    eval_records = fr_test + mediqa_test + medquad_eval
    stats = {
        "frenchmedmcqa_test": len(fr_test),
        "mediqa_test": len(mediqa_test),
        "medquad_reserve": len(medquad_eval),
        "total": len(eval_records),
    }
    return eval_records, stats


def split_sft(sft: list[dict], val_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Découpe le SFT en train/validation stratifié (langue × source)."""
    rng = random.Random(SEED)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in sft:
        groups[(r["lang"], r["source"])].append(r)

    train: list[dict] = []
    val: list[dict] = []
    for group in groups.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_all() -> dict[str, Any]:
    """Exécute la sélection SFT/DPO/éval + écrit les fichiers intermédiaires."""
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    sft, sft_stats, medquad_eval = select_sft()
    dpo, dpo_stats = select_dpo()
    eval_records, eval_stats = select_clinical_eval(medquad_eval)

    write_jsonl(sft, SELECTED_DIR / "sft.jsonl")
    write_jsonl(dpo, SELECTED_DIR / "dpo.jsonl")
    write_jsonl(eval_records, SELECTED_DIR / "clinical_eval.jsonl")

    manifest = {"sft": sft_stats, "dpo": dpo_stats, "clinical_eval": eval_stats}
    (SELECTED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Sélection : SFT=%d, DPO=%d, éval clinique=%d", len(sft), len(dpo), len(eval_records))
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all()
