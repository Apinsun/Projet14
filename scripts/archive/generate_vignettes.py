#!/usr/bin/env python3
"""Génère des vignettes patient (Phase 2) avec 3 prompts systèmes différents.

Usage :
    poetry run python scripts/generate_vignettes.py --n 15 --seed 42
"""

from __future__ import annotations

import argparse
import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.generate import generate_fact_sheets
from triage_agent.data.vignette import generate_vignette, parse_json_vignette

PROMPTS: dict[str, str] = {
    "A_simple": (
        "Tu es un patient qui se présente aux urgences. À partir des informations "
        "fournies, rédige uniquement ce que le patient dirait spontanément à l'agent "
        "de triage. N'ajoute rien qui ne soit pas fourni."
    ),
    "B_realiste": (
        "Tu joues le rôle d'un patient aux urgences. Rédige son message d'ouverture "
        "en français courant, avec un ton réaliste : langage simple, inquiétude "
        "proportionnelle à la gravité, éventuellement hésitations. Utilise uniquement "
        "les informations fournies, n'invente aucun symptôme, aucune constante, aucun "
        "diagnostic."
    ),
    "C_json": (
        "Tu génères des données d'entraînement médicales. Transforme les informations "
        "fournies en un message de patient en français. Réponds UNIQUEMENT avec un "
        "objet JSON : {\"vignette\": \"...\"}. N'invente aucun élément absent."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Génération de vignettes patient (Phase 2)")
    ap.add_argument("--n", type=int, default=15, help="Nombre de fact sheets par prompt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    facts = generate_fact_sheets(args.n, args.seed)
    out = PROCESSED_DIR / "triage" / "vignettes.jsonl"

    with out.open("w", encoding="utf-8") as fh:
        for name, prompt in PROMPTS.items():
            print(f"\n{'=' * 90}\nPROMPT {name}\n{'=' * 90}")
            for fact in facts:
                raw = generate_vignette(fact, prompt)
                vignette = parse_json_vignette(raw)
                rec = {
                    "case_id": fact["case_id"],
                    "prompt": name,
                    "true_level": fact["true_level"],
                    "vignette": vignette,
                    "fact": fact,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if fact["case_id"].endswith(("1", "5", "9")):
                    print(f"\n--- {fact['case_id']} | niveau {fact['true_level']} | "
                          f"{fact['motif']} ---")
                    print("VIGNETTE :", vignette[:400])
            fh.flush()

    print(f"\nRésultats écrits dans {out}")


if __name__ == "__main__":
    main()
