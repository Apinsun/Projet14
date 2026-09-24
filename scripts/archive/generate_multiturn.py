#!/usr/bin/env python3
"""Génère le dataset multi-tours avec le LLM fort (27B), fiche déterministe.

Le patient révèle son symptôme dès l'ouverture, puis les faits au fil des questions.
La fiche (niveau) est calculée par la règle FRENCH ; le LLM ne rédige que la langue
naturelle (<think>, questions, réponses, explication).

Usage :
    poetry run python scripts/generate_multiturn.py --limit 15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.dialogue_llm import (
    MODEL,
    build_reveal_schedule,
    generate_dialogue_llm,
    is_third_person_symptom,
)
from triage_agent.data.generate import generate_balanced_fact_sheets, load_rules


def _motif_label(rules: dict, fact: dict) -> str:
    return rules["categories"][fact["category"]]["motifs"][fact["motif"]]["label"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Génération du dataset multi-tours (LLM 27B)")
    ap.add_argument("--per-level", type=int, default=12, help="fact sheets par niveau (pool source)")
    ap.add_argument("--limit", type=int, default=0, help="nombre de dialogues à générer (0 = tous)")
    ap.add_argument("--min-qa", type=int, default=2, help="nombre minimal de questions/réponses par dialogue")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "sft_multiturn_llm.jsonl"))
    args = ap.parse_args()

    rules = load_rules()
    facts = generate_balanced_fact_sheets(args.per_level, args.seed)

    # Sélection des fact sheets avec assez de matière questionnable (multi-tours riche).
    selected: list[tuple[dict, list[tuple[str, str]]]] = []
    for fact in facts:
        if is_third_person_symptom(fact["rule"].get("symptom", "")):
            continue
        cr = fact["patient_observable"]["can_report"]
        rng = random.Random(hash(fact["case_id"]) & 0xFFFFFFFF)
        schedule = build_reveal_schedule(cr, rng)
        if len(schedule) >= args.min_qa:
            selected.append((fact, schedule))
        if args.limit and len(selected) >= args.limit:
            break

    print(f"{len(selected)} dialogues à générer (pool {len(facts)} fact sheets).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"llm": 0, "fallback": 0, "reasons": {}, "fiche": {"llm_ok": 0, "injectee": 0}}
    with out_path.open("w", encoding="utf-8") as fh:
        for i, (fact, schedule) in enumerate(selected):
            rule = fact["rule"]
            label = _motif_label(rules, fact)
            messages, s = generate_dialogue_llm(fact, rule, rules["levels"], label, args.model, schedule)
            stats[s["method"]] += 1
            if s["method"] == "fallback":
                r = s.get("reason", "?")
                stats["reasons"][r] = stats["reasons"].get(r, 0) + 1
            else:
                f = s.get("fiche", "?")
                stats["fiche"][f] = stats["fiche"].get(f, 0) + 1

            fh.write(json.dumps({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_triage_multiturn_llm",
                    "true_level": fact["true_level"],
                    "case_id": fact["case_id"],
                    "method": s["method"],
                },
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 3 == 0:
                print(f"  {i + 1}/{len(selected)} ...")
                fh.flush()

    print(f"Terminé. {len(selected)} dialogues -> {out_path}")
    print(f"LLM : {stats['llm']} | fallback : {stats['fallback']} | raisons : {stats['reasons']}")
    print(f"Fiches : LLM conformes {stats['fiche']['llm_ok']} | injectées {stats['fiche']['injectee']}")


if __name__ == "__main__":
    main()
