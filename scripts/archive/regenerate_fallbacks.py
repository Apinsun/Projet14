#!/usr/bin/env python3
"""Régénère les dialogues tombés en repli d'un batch multi-tours.

Utile quand Ollama a été indisponible (repli massif « erreur 404 ») : on re-tente le
LLM sur les dialogues scriptés. Les cas à la 3e personne sont laissés en repli (ils
seront filtrés ensuite par ``finalize_multiturn.py``).

Usage :
    poetry run python scripts/regenerate_fallbacks.py \
      --input data/processed/triage/sft_multiturn_batch2.jsonl \
      --output data/processed/triage/sft_multiturn_batch2_v2.jsonl \
      --per-level 200 --seed 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage_agent.data.dialogue_llm import MODEL, generate_dialogue_llm, is_third_person_symptom
from triage_agent.data.generate import generate_balanced_fact_sheets, load_rules


def _motif_label(rules: dict, fact: dict) -> str:
    return rules["categories"][fact["category"]]["motifs"][fact["motif"]]["label"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Régénère les replis d'un batch multi-tours")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-level", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="repli max à régénérer (0 = tous)")
    args = ap.parse_args()

    rules = load_rules()
    facts = generate_balanced_fact_sheets(args.per_level, args.seed)
    fact_by_id = {f["case_id"]: f for f in facts}

    recs = [json.loads(line) for line in Path(args.input).open(encoding="utf-8")]

    n_llm = n_fallback = n_skipped = 0
    reasons: dict[str, int] = {}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for rec in recs:
            if rec["metadata"]["method"] != "fallback" or (args.limit and n_llm + n_fallback >= args.limit):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            cid = rec["metadata"]["case_id"]
            fact = fact_by_id.get(cid)
            if fact is None or is_third_person_symptom(fact["rule"].get("symptom", "")):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")  # laissé en repli
                n_skipped += 1
                continue

            label = _motif_label(rules, fact)
            messages, stats = generate_dialogue_llm(fact, fact["rule"], rules["levels"], label, args.model)
            rec["messages"] = messages
            rec["metadata"]["method"] = stats["method"]
            if stats["method"] == "llm":
                n_llm += 1
            else:
                n_fallback += 1
                r = stats.get("reason", "?")
                reasons[r] = reasons.get(r, 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            if (n_llm + n_fallback) % 3 == 0:
                print(f"  régénérés {n_llm + n_fallback} ...", flush=True)

    print(f"Régénéré : {n_llm} via LLM | {n_fallback} encore en repli | {n_skipped} laissés (3e personne)")
    print(f"Raisons de repli restant : {reasons}")
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
