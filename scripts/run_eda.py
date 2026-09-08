#!/usr/bin/env python3
"""Analyse exploratoire des datasets bruts.

Génère ``reports/eda_results.json`` (résumé machine-readable) et affiche un
résumé lisible. Le rapport Markdown ``reports/01_analyse_datasets.md`` s'appuie
sur ces résultats.

Usage :
    poetry run python scripts/run_eda.py
"""

from __future__ import annotations

import json

from triage_agent.data.config import REPO_ROOT
from triage_agent.data.explore import summarize_all


def _dump_text(results: dict) -> None:
    for name, r in results.items():
        print("=" * 80)
        print(f"[{name}]  hf={r['hf_id']}  langue={r['language']}  rôle={r['role']}")
        print(f"  splits        : {r['splits']}")
        print(f"  colonnes      : {list(r['columns'])}")
        print(f"  valeurs manqu.: {r['missing']}")
        if "duplicates" in r:
            d = r["duplicates"]
            print(
                f"  doublons      : {d['duplicated_rows']} dupliqués / "
                f"{d['total_rows']} lignes ({d['unique_values']} uniques)"
            )
        if "language_distribution" in r:
            print(f"  langues       : {r['language_distribution']}")
        print("  longueurs (chars) :")
        for col, st in r["lengths"].items():
            print(f"    {col:20s} mean={st['mean']}  median={st['median']}  "
                  f"min={st['min']}  max={st['max']}")
        if r.get("html_tags"):
            print(f"  balises HTML  : {r['html_tags']}")
        if r.get("specific"):
            print(f"  spécifique    : {json.dumps(r['specific'], ensure_ascii=False)}")


def main() -> None:
    results = summarize_all()

    out = REPO_ROOT / "reports" / "eda_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats écrits dans {out}\n")

    _dump_text(results)


if __name__ == "__main__":
    main()
