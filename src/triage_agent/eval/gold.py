"""Mappings des labels gold → plage FRENCH acceptable + construction des records.

Chaque label gold (4 ou 3 niveaux) devient une **plage FRENCH acceptable** :

- ``[1, 3]``  = « urgences maintenant » (le gold ne distingue pas FRENCH 1/2/3) ;
- ``[4, 4]``  = semi-urgent (24–48 h) ;
- ``[5, 5]``  = bénin ;
- ``[3, 4]``  = split label (désaccord entre médecins) → les 2 bornes acceptées.

``binary_urgent`` (pour la métrique binaire de sécurité) vaut ``True`` si **toutes**
les réponses acceptables sont urgentes (borne haute ≤ 3).
"""

from __future__ import annotations

import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.fiche import SYSTEM_PROMPT

RAMASWAMY_RANGE: dict[str, list[int]] = {
    "A": [5, 5],
    "A/B": [5, 5],
    "B": [5, 5],
    "B/C": [4, 5],
    "C": [4, 4],
    "C/D": [3, 4],
    "D": [1, 3],
}

LEVINE_RANGE: dict[str, list[int]] = {
    "Emergent": [1, 3],
    "1 day": [4, 4],
    "1 week": [5, 5],
    "Self-care": [5, 5],
}

IYAWOBENCH_RANGE: dict[str, list[int]] = {
    "REFER_NOW": [1, 3],
    "REFER_TODAY": [4, 4],
    "TREAT_HERE": [5, 5],
}


def make_record(
    source: str,
    case_id: str,
    domain: str,
    patient_msg: str,
    gold_label: str,
    gold_range: list[int],
) -> dict:
    """Construit un record d'évaluation au format ``{"messages", "metadata"}``."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_msg},
        ],
        "metadata": {
            "source": source,
            "case_id": case_id,
            "domain": domain,
            "gold_label_original": gold_label,
            "gold_range": gold_range,
            "binary_urgent": gold_range[1] <= 3,
        },
    }


def load_few_shot_examples() -> list[dict]:
    """Charge 2 exemples few-shot (1 niveau 1, 1 niveau 5) depuis les vignettes SFT.

    Retourne les tours ``user``/``assistant`` (sans le system) à insérer entre le
    system prompt d'évaluation et le message patient réel.
    """
    path = PROCESSED_DIR / "triage" / "sft_vignettes.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    picked: dict[int, dict] = {}
    for rec in records:
        lvl = rec["metadata"]["true_level"]
        if lvl in (1, 5) and lvl not in picked:
            picked[lvl] = rec
    examples: list[dict] = []
    for lvl in (1, 5):
        for msg in picked[lvl]["messages"]:
            if msg["role"] in ("user", "assistant"):
                examples.append({"role": msg["role"], "content": msg["content"]})
    return examples
