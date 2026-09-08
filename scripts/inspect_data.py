#!/usr/bin/env python3
"""Affiche des enregistrements aléatoires des jeux finaux pour inspection manuelle.

Usage :
    poetry run python scripts/inspect_data.py sft_train 3        # 3 échantillons
    poetry run python scripts/inspect_data.py dpo_train 2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap

from triage_agent.data.config import PROCESSED_DIR

FINAL = PROCESSED_DIR / "final"


def _wrap(text: str, width: int = 110) -> str:
    """Enveloppe le texte en préservant les retours à la ligne existants."""
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, width) or [""])
    return "\n".join(lines)


def show(records: list[dict], n: int) -> None:
    for r in random.sample(records, min(n, len(records))):
        print("=" * 100)
        meta = r.get("metadata", {})
        print(
            f"lang={r.get('lang')} | source={r.get('source')} | split={meta.get('split')} "
            f"| id={r.get('id')}"
        )
        if meta.get("question_type"):
            print(f"question_type={meta['question_type']} | topic={meta.get('topic')}")
        print("-" * 100)
        if r.get("role") == "dpo":
            print("[prompt]  ", _wrap(r["prompt"]))
            print("[chosen]  ", _wrap(r["chosen"]))
            print("[rejected]", _wrap(r["rejected"]))
        else:
            for msg in r["messages"]:
                print(f"[{msg['role']}]")
                print(_wrap(msg["content"]))
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspection manuelle des jeux finaux")
    ap.add_argument("dataset", choices=["sft_train", "sft_val", "dpo_train", "clinical_eval"])
    ap.add_argument("n", type=int, nargs="?", default=3, help="Nombre d'échantillons (défaut 3)")
    ap.add_argument("--seed", type=int, default=None, help="Seed aléatoire (reproductible)")
    args = ap.parse_args()

    random.seed(args.seed)
    path = FINAL / f"{args.dataset}.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    show(records, args.n)


if __name__ == "__main__":
    main()
