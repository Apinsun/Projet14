#!/usr/bin/env python3
"""Réécrit les symptômes « conscients » à la 3e personne en « je » via le LLM 27B.

Les symptômes décrivant un patient **conscient** (mais tournés au « il ») sont
reformulés à la 1re personne par le LLM. Les symptômes « inconscients » (arrêt
cardiaque, coma…) sont laissés tels quels : ils seront traités en mode « tiers ».

Usage :
    poetry run python scripts/rewrite_symptoms.py            # applique
    poetry run python scripts/rewrite_symptoms.py --dry-run  # aperçu sans écrire
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from triage_agent.data.dialogue_llm import (
    _NO_THINK_STOP,
    MODEL,
    is_inconscient_symptom,
    is_third_person_symptom,
)
from triage_agent.data.vignette import ollama_chat

RULES_PATH = Path("src/triage_agent/data/french_rules.json")

SYSTEM = (
    "Tu réécris un symptôme médical rapporté aux urgences, en le mettant à la première "
    "personne (« je »), comme si le patient lui-même le décrivait. Conserve exactement le "
    "sens clinique, le niveau de gravité et le vocabulaire. Réponds uniquement avec la "
    "phrase réécrite, sans guillemets, sans commentaire."
)

_FIRST_PERSON = ("je ", "j'", "j’", "mon ", "ma ", "mes ", "nous ", "le bébé", "notre ")


def _is_first_person(text: str) -> bool:
    return text.strip().lower().startswith(_FIRST_PERSON)


def main() -> None:
    ap = argparse.ArgumentParser(description="Réécrit les symptômes 3e personne en « je »")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    bak = RULES_PATH.with_suffix(".json.bak")
    shutil.copy(RULES_PATH, bak)

    n = n_ok = 0
    for cat in data["categories"].values():
        for m in cat["motifs"].values():
            for rule in m["rules"]:
                sym = rule.get("symptom", "")
                if not (is_third_person_symptom(sym) and not is_inconscient_symptom(sym)):
                    continue
                rew = ollama_chat(
                    args.model, SYSTEM, f"Symptôme : {sym}\nRéécrit en « je » :",
                    temperature=0.4, timeout=300, think=False, stop=_NO_THINK_STOP,
                ).strip().strip('"').strip()
                if _is_first_person(rew) and not is_third_person_symptom(rew):
                    print(f"  ✓ {sym}\n    → {rew}\n")
                    if not args.dry_run:
                        rule["symptom"] = rew
                    n_ok += 1
                else:
                    print(f"  ✗ échec (conservé) : {sym}  →  {rew}\n")
                n += 1

    print(f"{n_ok}/{n} symptômes réécrits.")
    if not args.dry_run:
        RULES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Sauvegardé : {RULES_PATH} (backup : {bak})")


if __name__ == "__main__":
    main()
