#!/usr/bin/env python3
"""Génère des fact sheets de triage (Phase 1) et affiche des exemples.

Usage :
    poetry run python scripts/generate_fact_sheets.py 20 --seed 42
"""

from __future__ import annotations

import argparse
import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.generate import generate_fact_sheets


def _show(fact: dict) -> None:
    print("=" * 100)
    print(f"{fact['case_id']} | {fact['motif']} | âge {fact['age']} {fact['sex']} | niveau {fact['level_label']}")
    print(f"  justification : {fact['justification']}")
    print(f"  signes objectifs : {fact['signs_objectifs']}")
    print(f"  constantes : {fact['vital_signs']}")
    print(f"  antécédents : {fact['medical_history']} | traitements : {fact['treatments']}")
    po = fact["patient_observable"]
    cr = po["can_report"]
    print(f"  patient peut dire : {cr['symptoms']} | douleur {cr['pain_scale']} | durée {cr['duration']}")
    print(f"  patient NE peut PAS dire : {po['cannot_report']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Génération de fact sheets de triage")
    ap.add_argument("n", type=int, default=10, help="Nombre de fact sheets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    facts = generate_fact_sheets(args.n, args.seed)

    out = PROCESSED_DIR / "triage" / "fact_sheets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"{len(facts)} fact sheets écrites dans {out}\n")

    for f in facts[:5]:
        _show(f)
        print()


if __name__ == "__main__":
    main()
