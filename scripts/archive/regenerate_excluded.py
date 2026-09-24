#!/usr/bin/env python3
"""Régénère les dialogues exclus (symptômes 3e personne) avec le bon mode.

- « conscients » (symptômes réécrits en « je ») → dialogue patient normal ;
- « inconscients » → dialogue « tiers » (un proche parle).

Les case_ids exclus sont identifiés par différence : présents dans les batchs mais
absents du dataset final (filtrés par ``finalize_multiturn.py``).

Usage :
    poetry run python scripts/regenerate_excluded.py \
      --final data/processed/triage/sft_multiturn_llm.jsonl \
      --batch data/processed/triage/sft_multiturn_batch1.jsonl \
              data/processed/triage/sft_multiturn_batch2_v2.jsonl \
      --per-level 30 --per-level 200 --seed 43 --seed 44 \
      --out data/processed/triage/sft_multiturn_excluded.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from triage_agent.data.dialogue_llm import (
    MODEL,
    build_reveal_schedule,
    generate_dialogue_llm,
    is_inconscient_symptom,
)
from triage_agent.data.generate import generate_balanced_fact_sheets, load_rules


def _motif_label(rules: dict, fact: dict) -> str:
    return rules["categories"][fact["category"]]["motifs"][fact["motif"]]["label"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Régénère les dialogues exclus (3e personne)")
    ap.add_argument("--final", required=True)
    ap.add_argument("--batch", nargs="+", required=True)
    ap.add_argument("--per-level", type=int, action="append", required=True)
    ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rules = load_rules()

    final_ids = {json.loads(line)["metadata"]["case_id"] for line in Path(args.final).open(encoding="utf-8")}
    batch_ids: list[str] = []
    for b in args.batch:
        batch_ids += [json.loads(line)["metadata"]["case_id"] for line in Path(b).open(encoding="utf-8")]

    seen: set[str] = set()
    excluded = [c for c in batch_ids if c not in final_ids and not (c in seen or seen.add(c))]

    facts = []
    for pl, sd in zip(args.per_level, args.seed):
        facts += generate_balanced_fact_sheets(pl, sd)
    fact_by_id = {f["case_id"]: f for f in facts}

    if args.limit:
        excluded = excluded[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_llm = n_fallback = 0
    reasons: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fh:
        for cid in excluded:
            fact = fact_by_id.get(cid)
            if fact is None:
                continue
            rule = fact["rule"]
            label = _motif_label(rules, fact)
            third_party = is_inconscient_symptom(rule.get("symptom", ""))
            schedule = build_reveal_schedule(
                fact["patient_observable"]["can_report"], random.Random(), third_party
            )
            messages, stats = generate_dialogue_llm(
                fact, rule, rules["levels"], label, args.model, schedule, third_party
            )
            fh.write(json.dumps({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_triage_multiturn_llm",
                    "true_level": fact["true_level"],
                    "case_id": fact["case_id"],
                    "method": stats["method"],
                    "third_party": third_party,
                },
            }, ensure_ascii=False) + "\n")
            if stats["method"] == "llm":
                n_llm += 1
            else:
                n_fallback += 1
                r = stats.get("reason", "?")
                reasons[r] = reasons.get(r, 0) + 1
            fh.flush()
            if (n_llm + n_fallback) % 5 == 0:
                print(f"  {n_llm + n_fallback}/{len(excluded)} ...", flush=True)

    print(f"Régénéré : {n_llm} via LLM | {n_fallback} repli | raisons {reasons}")
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
