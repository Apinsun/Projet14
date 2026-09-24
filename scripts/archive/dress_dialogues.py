#!/usr/bin/env python3
"""Habille les dialogues scriptés avec le LLM + vérification anti-incohérence.

Usage :
    poetry run python scripts/dress_dialogues.py --limit 10
"""

from __future__ import annotations

import argparse
import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.dress import dress_dialogue


def main() -> None:
    ap = argparse.ArgumentParser(description="Habillage LLM des dialogues")
    ap.add_argument("--limit", type=int, default=0, help="Limiter à N dialogues (0 = tous)")
    args = ap.parse_args()

    src = PROCESSED_DIR / "triage" / "sft_dialogues.jsonl"
    dst = PROCESSED_DIR / "triage" / "sft_dialogues_dressed.jsonl"

    records = [json.loads(line) for line in src.open(encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]

    total_dressed = 0
    total_kept = 0
    reasons: dict[str, int] = {}

    with dst.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            messages, stats = dress_dialogue(rec["messages"])
            total_dressed += stats["dressed"]
            total_kept += stats["kept"]
            for k, v in stats["reasons"].items():
                reasons[k] = reasons.get(k, 0) + v

            fh.write(json.dumps({**rec, "messages": messages}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(records)} ...")
                fh.flush()

    print(f"Terminé. {len(records)} dialogues.")
    print(f"Réponses habillées : {total_dressed} | conservées (scriptées) : {total_kept}")
    print(f"Motifs de conservation : {reasons}")
    print(f"Sortie : {dst}")


if __name__ == "__main__":
    main()
