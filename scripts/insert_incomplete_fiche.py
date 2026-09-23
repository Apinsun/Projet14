#!/usr/bin/env python3
"""Insère une fiche INCOMPLÈTE à chaque tour intermédiaire des dialogues multi-tours.

Fiche unique : le SI se charge de créer ou mettre à jour le dossier (pas de
``update``/``finalize``). Aux tours intermédiaires, la fiche reflète l'état partiel :
``priority_range [1, niveau_final]`` (borne prudente), ``missing_info`` = champs que
l'agent n'a pas encore demandés, ``confidence`` basse. Le tour final garde sa fiche
complète (déterministe).

L'ordre des champs restants est dérivé des QUESTIONS de l'agent (pas d'un seed, car le
générateur utilisait ``hash()`` non reproductible). Le ``niveau`` n'est jamais deviné :
la borne haute vient du ``true_level`` du fact sheet (re-dérivé par ``case_id``).

Usage :
    poetry run python scripts/insert_incomplete_fiche.py --limit 200 --out .../subset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.generate import generate_balanced_fact_sheets

POOLS = [(30, 43), (200, 44), (12, 43)]

_QUESTION_KEYWORDS: dict[str, list[str]] = {
    "antécédents": ["antécédent", "maladie", "problème de santé", "santé"],
    "traitements": ["traitement", "médicament", "prenez", "prend"],
    "durée": ["depuis combien", "depuis quand", "combien de temps", "durée"],
    "température": ["température", "degrés", "fièvre"],
    "intensité": ["0 à 10", "échelle", "douleur", "intensité"],
}


def build_lookup() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for per_level, seed in POOLS:
        for f in generate_balanced_fact_sheets(per_level, seed):
            lookup.setdefault(f["case_id"], f)
    return lookup


def question_field(question: str) -> str | None:
    """Devine le champ ciblé par une question de l'agent."""
    q = question.lower()
    for field, kws in _QUESTION_KEYWORDS.items():
        if any(k in q for k in kws):
            return field
    return None


def _fiche_json(true_level: int, remaining: list[str], summary: str) -> str:
    fiche = {
        "name": "finalize_triage",
        "arguments": {
            "priority_range": [1, true_level],
            "red_flags": [],
            "missing_info": remaining,
            "confidence": 0.4,
            "summary": summary,
        },
    }
    return "<FICHE>\n" + json.dumps(fiche, ensure_ascii=False) + "\n</FICHE>"


def insert(content: str, fiche_block: str) -> str:
    marker = "</think>"
    idx = content.find(marker)
    if idx == -1:
        return content
    return content[: idx + len(marker)] + "\n" + fiche_block + content[idx + len(marker):]


def main() -> None:
    ap = argparse.ArgumentParser(description="Insère une fiche incomplète à chaque tour intermédiaire")
    ap.add_argument("--limit", type=int, default=0, help="Nb de dialogues à traiter (0 = tous)")
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "sft_multiturn_fiche.jsonl"))
    ap.add_argument("--input", default=str(PROCESSED_DIR / "triage" / "sft_multiturn_full.jsonl"))
    args = ap.parse_args()

    lookup = build_lookup()
    src = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for line in src.open(encoding="utf-8"):
            rec = json.loads(line)
            fact = lookup.get(rec["metadata"]["case_id"])
            if fact is None:
                skipped += 1
                continue
            true_level = fact["true_level"]
            summary = fact["rule"].get("symptom", "évaluation en cours").strip().rstrip(".")

            msgs = rec["messages"]
            # ordre des champs dérivé des questions posées aux tours intermédiaires
            order: list[str | None] = []
            for i in range(2, len(msgs) - 1, 2):
                question = msgs[i]["content"].split("</think>")[-1].strip()
                order.append(question_field(question))

            # injection aux tours intermédiaires (question k → il reste les champs order[k:])
            for k, i in enumerate(range(2, len(msgs) - 1, 2)):
                remaining = [f for f in order[k:] if f]
                msgs[i]["content"] = insert(msgs[i]["content"],
                                            _fiche_json(true_level, remaining, summary))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if args.limit and done >= args.limit:
                break

    print(f"{done} dialogues traités | {skipped} sans fact sheet -> {out}")


if __name__ == "__main__":
    main()
