#!/usr/bin/env python3
"""Génère le dataset de triage complet (vignettes mono-tour + dialogues multi-tours).

Usage :
    poetry run python scripts/generate_dataset.py --vignettes-per-level 60 --dialogues-per-level 60
"""

from __future__ import annotations

import argparse
import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.dialogue import build_dialogue
from triage_agent.data.fiche import SYSTEM_PROMPT, explanation, fiche_block
from triage_agent.data.generate import generate_balanced_fact_sheets, load_rules
from triage_agent.data.vignette import generate_vignette, parse_json_vignette

VIGNETTE_PROMPT = (
    "Tu joues le rôle d'un patient aux urgences. Rédige son message d'ouverture "
    "en français courant, avec un ton réaliste : langage simple, inquiétude "
    "proportionnelle à la gravité, éventuellement hésitations. Utilise uniquement "
    "les informations fournies, n'invente aucun symptôme, aucune constante, aucun "
    "diagnostic."
)


def _motif_label(rules: dict, fact: dict) -> str:
    return rules["categories"][fact["category"]]["motifs"][fact["motif"]]["label"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Génération du dataset de triage")
    ap.add_argument("--vignettes-per-level", type=int, default=60)
    ap.add_argument("--dialogues-per-level", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rules = load_rules()
    facts = generate_balanced_fact_sheets(args.vignettes_per_level, args.seed)
    print(f"{len(facts)} fact sheets équilibrées générées.")

    out_dir = PROCESSED_DIR / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    vignettes_path = out_dir / "sft_vignettes.jsonl"
    dialogues_path = out_dir / "sft_dialogues.jsonl"

    n_vignettes = 0
    n_dialogues = 0

    with vignettes_path.open("w", encoding="utf-8") as fv, dialogues_path.open("w", encoding="utf-8") as fd:
        for i, fact in enumerate(facts):
            rule = fact["rule"]
            label = _motif_label(rules, fact)
            fiche = fiche_block(fact, rule, rules["levels"])
            expl = explanation(rule)

            # --- Vignette (mono-tour) : message patient -> fiche ---
            try:
                raw = generate_vignette(fact, VIGNETTE_PROMPT)
                vignette = parse_json_vignette(raw)
            except Exception as exc:  # résilience : ne pas tout perdre sur un échec
                print(f"  [warn] échec vignette {fact['case_id']} : {exc} -> fallback symptom")
                vignette = rule["symptom"]
            fv.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": vignette},
                    {"role": "assistant", "content": f"{fiche}\n{expl}"},
                ],
                "metadata": {
                    "source": "synthetic_triage_vignette",
                    "true_level": fact["true_level"],
                    "case_id": fact["case_id"],
                },
            }, ensure_ascii=False) + "\n")
            n_vignettes += 1

            # --- Dialogue (multi-tours, scripté) : questionnaire + fiche ---
            dialogue = build_dialogue(fact, rule, rules["levels"], label)
            fd.write(json.dumps({
                "messages": dialogue,
                "metadata": {
                    "source": "synthetic_triage_dialogue",
                    "true_level": fact["true_level"],
                    "case_id": fact["case_id"],
                },
            }, ensure_ascii=False) + "\n")
            n_dialogues += 1

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(facts)} ...")
                fv.flush()
                fd.flush()

    print(f"Vignettes (mono-tour) : {n_vignettes} -> {vignettes_path}")
    print(f"Dialogues (multi-tours) : {n_dialogues} -> {dialogues_path}")


if __name__ == "__main__":
    main()
