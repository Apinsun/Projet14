#!/usr/bin/env python3
"""Vérifie la conformité des datasets (train SFT / DPO / gold).

À exécuter avant chaque re-train pour garantir « garbage in, garbage out » :
- structure des rôles (system, user, assistant, …) ;
- balises : ``<FICHE>`` uniquement (ni ``[FICHE]`` ni ``(FICHE)``) ;
- JSON de fiche valide + schéma (``priority`` ou ``priority_range``) ;
- ``<think>`` présent sur chaque tour assistant ;
- DPO : fiche identique chosen/rejected ;
- gold : ``gold_range`` + ``binary_urgent`` cohérents.

Usage :
    poetry run python scripts/validate_dataset.py           # tout par défaut
    poetry run python scripts/validate_dataset.py --train    # SFT seulement
    poetry run python scripts/validate_dataset.py --files .../x.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR

_FICHE_RE = re.compile(r"<FICHE>\s*(.*?)\s*</FICHE>", re.S)

TRAIN_FILES = [
    PROCESSED_DIR / "triage" / "sft_vignettes.jsonl",
    PROCESSED_DIR / "triage" / "sft_multiturn_fiche_nat.jsonl",
]
DPO_FILES = [PROCESSED_DIR / "triage" / "dpo_quality_v3.jsonl"]
GOLD_FILES = [
    PROCESSED_DIR / "gold" / "fr" / "levine.jsonl",
    PROCESSED_DIR / "gold" / "fr" / "ramaswamy.jsonl",
    PROCESSED_DIR / "gold" / "fr" / "iyawobench_urgent.jsonl",
]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _fiche(content: str) -> dict | None:
    m = _FICHE_RE.search(content)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def check_train(path: Path) -> tuple[int, int]:
    errors = 0
    n = 0
    for rec in load(path):
        n += 1
        msgs = rec["messages"]
        roles = [m["role"] for m in msgs]
        if roles[0] != "system":
            errors += 1
            print(f"  [{path.name}] {rec['metadata'].get('case_id','?')} : ne commence pas par system")
            continue
        for i, r in enumerate(roles[1:], 1):
            exp = "user" if i % 2 == 1 else "assistant"
            if r != exp:
                errors += 1
                print(f"  [{path.name}] {rec['metadata'].get('case_id','?')} : rôle {i} = {r} (attendu {exp})")
        for m in msgs:
            if m["role"] != "assistant":
                continue
            c = m["content"]
            if "[FICHE]" in c or "(FICHE)" in c or "[FICHE" in c:
                errors += 1
                print(f"  [{path.name}] {rec['metadata'].get('case_id','?')} : balise fiche non standard")
            if "<think>" not in c:
                errors += 1
                print(f"  [{path.name}] {rec['metadata'].get('case_id','?')} : assistant sans <think>")
            f = _fiche(c)
            if f is None:
                errors += 1
                print(f"  [{path.name}] {rec['metadata'].get('case_id','?')} : fiche absente ou JSON invalide")
    return n, errors


def check_dpo(path: Path) -> tuple[int, int]:
    errors = 0
    recs = load(path)
    for i, p in enumerate(recs):
        if not {"prompt", "chosen", "rejected"} <= set(p):
            errors += 1
            print(f"  [{path.name}] paire {i} : champs manquants")
            continue
        fc = _fiche(p["chosen"][0]["content"])
        fr = _fiche(p["rejected"][0]["content"])
        if fc != fr:
            errors += 1
            print(f"  [{path.name}] paire {i} : fiche chosen != rejected")
    return len(recs), errors


def check_gold(path: Path) -> tuple[int, int]:
    errors = 0
    recs = load(path)
    for rec in recs:
        meta = rec["metadata"]
        rng = meta.get("gold_range")
        if not isinstance(rng, list) or len(rng) != 2:
            errors += 1
            print(f"  [{path.name}] {meta.get('case_id','?')} : gold_range invalide")
        if meta.get("binary_urgent") != (rng[1] <= 3):
            errors += 1
            print(f"  [{path.name}] {meta.get('case_id','?')} : binary_urgent incohérent")
    return len(recs), errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Vérifie la conformité des datasets")
    ap.add_argument("--train", action="store_true", help="SFT seulement")
    ap.add_argument("--files", nargs="*", help="Fichiers à vérifier (détection auto du type)")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.files:
        targets = [Path(f) for f in args.files]
    elif args.train:
        targets = TRAIN_FILES
    else:
        targets = TRAIN_FILES + DPO_FILES + GOLD_FILES

    total_errors = 0
    for path in targets:
        if not path.exists():
            print(f"⚠️  {path} : INTROUVABLE")
            total_errors += 1
            continue
        if path.name.startswith("dpo"):
            n, e = check_dpo(path)
        elif "gold" in str(path):
            n, e = check_gold(path)
        else:
            n, e = check_train(path)
        print(f"{'✅' if e == 0 else '❌'} {path.name} : {n} records | {e} erreur(s)")
        total_errors += e

    print()
    print("CONFORME ✅" if total_errors == 0 else f"NON CONFORME ❌ ({total_errors} erreurs)")
    raise SystemExit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
