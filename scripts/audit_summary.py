#!/usr/bin/env python3
"""Audit de traçabilité : résumé du journal d'audit (logs/audit.jsonl).

Usage :
    poetry run python scripts/audit_summary.py [chemin-du-journal]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = {"_ts", "conversation_id", "messages", "raw_output", "think", "fiche", "parse_ok"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Résumé du journal d'audit")
    ap.add_argument("path", nargs="?", default="logs/audit.jsonl")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"❌ aucun journal d'audit : {p}")
        return 1

    records = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    convs = Counter(r.get("conversation_id") for r in records)
    parsed = sum(1 for r in records if r.get("parse_ok"))
    incomplete = sum(1 for r in records if not REQUIRED <= set(r))
    with_think = sum(1 for r in records if r.get("think"))

    print("=== Audit de traçabilité ===")
    print(f"échanges tracés        : {len(records)}")
    print(f"conversations distinctes: {len(convs)}")
    print(f"fiches bien formées    : {parsed}/{len(records)} "
          f"({100 * parsed / len(records):.1f} %)" if records else "—")
    print(f"échanges avec <think>  : {with_think}/{len(records)}")
    print(f"enregistrements incomplets : {incomplete}")
    if records:
        print(f"période                : {records[0]['_ts']} → {records[-1]['_ts']}")
        print("\n=== par conversation ===")
        for cid, n in convs.most_common():
            print(f"  {cid}: {n} échange(s)")

    if incomplete:
        print(f"\n⚠️  {incomplete} enregistrement(s) incomplet(s) — vérifier la traçabilité.")
        return 1
    print("\n✅ journal cohérent : chaque échange est tracé et complet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
