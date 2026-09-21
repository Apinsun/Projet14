#!/usr/bin/env python3
"""Fusionne les batchs multi-tours et exclut les cas à la 3e personne.

Les scénarios « patient inconscient / tiers » (symptôme à la 3e personne) ne sont pas
cohérents pour un dialogue patient en 1re personne : on les retire après coup.

Usage :
    poetry run python scripts/finalize_multiturn.py \
      --inputs batch1.jsonl batch2.jsonl \
      --per-level 30 --per-level 200 --seed 43 --seed 44 \
      --out sft_multiturn_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage_agent.data.dialogue_llm import is_third_person_symptom
from triage_agent.data.generate import generate_balanced_fact_sheets


def bad_case_ids(per_level: int, seed: int) -> set[str]:
    """Re-dérive les case_ids dont le symptôme est à la 3e personne (déterministe)."""
    facts = generate_balanced_fact_sheets(per_level, seed)
    return {f["case_id"] for f in facts if is_third_person_symptom(f["rule"].get("symptom", ""))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusionne et filtre les batchs multi-tours")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--per-level", type=int, action="append", required=True)
    ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.per_level) != len(args.seed) or len(args.per_level) != len(args.inputs):
        raise SystemExit("--inputs, --per-level et --seed doivent avoir la même longueur")

    bad: set[str] = set()
    for pl, sd in zip(args.per_level, args.seed):
        bad |= bad_case_ids(pl, sd)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = removed = 0
    with out.open("w", encoding="utf-8") as fh:
        for inp in args.inputs:
            for line in Path(inp).open(encoding="utf-8"):
                rec = json.loads(line)
                if rec["metadata"]["case_id"] in bad:
                    removed += 1
                    continue
                fh.write(line)
                kept += 1

    print(f"{kept} conservés | {removed} exclus (3e personne) -> {out}")


if __name__ == "__main__":
    main()
