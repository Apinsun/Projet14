#!/usr/bin/env python3
"""Construit un jeu gold « urgent » (métrique binaire de sécurité) depuis IyàwóBench.

IyàwóBench (200 vignettes de fièvre indifférenciée en soins primaires) n'avait été
écarté que par prudence (« constantes non-observables »). Or **les 100 cas REFER_NOW
portent tous un red flag patient-reportable** (convulsions, troubles de la conscience,
dyspnée, tirage, raideur de nuque) : le niveau d'urgence est donc dérivable sans les
constantes. Leur label est **externe** (indépendant de notre génération) → non circulaire.

Objectif : compléter notre gold, trop pauvre en cas urgents (18/87 → IC du binaire
sécurité à ±30 pts). On n'ajoute que des REFER_NOW (→ FRENCH [1, 3], `binary_urgent`).

Le message patient est en français naturel, **sans constantes** (non-reportables) ;
disease / constantes restent en métadonnées pour traçabilité.

Usage :
    poetry run python scripts/build_urgent_gold.py --n 48
"""

from __future__ import annotations

import argparse
import json

from triage_agent.data.config import PROCESSED_DIR, RAW_DIR
from triage_agent.eval.gold import IYAWOBENCH_RANGE, make_record

# Symptômes IyàwóBench → formulation patient française (détermini, sans invente).
FR_SYMPTOM: dict[str, str] = {
    "Fever": "de la fièvre",
    "Altered consciousness": "des troubles de la conscience",
    "Convulsions": "des convulsions",
    "Difficulty breathing": "des difficultés à respirer",
    "Chest indrawing": "un tirage respiratoire",
    "Stiff neck": "une raideur de la nuque",
    "Vomiting": "des vomissements",
    "Headache": "des maux de tête",
    "Fatigue": "une grande fatigue",
    "Abdominal pain": "des douleurs au ventre",
    "Diarrhoea": "de la diarrhée",
    "Loss of appetite": "une perte d'appétit",
    "Rash": "des boutons sur la peau",
    "Joint pain": "des douleurs articulaires",
}

# Symptômes qu'un patient ne peut pas rapporter lui-même → un proche parle (3e personne).
OBSERVER_SYMPTOMS = {"Altered consciousness", "Convulsions", "Chest indrawing"}


def _join(items: list[str]) -> str:
    """« a, b et c »."""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def build_fr_message(v: dict) -> str:
    """Message patient français naturel (reportable-seul, pas de constantes)."""
    age = v["patientAge"]
    unit = "an" if age <= 1 else "ans"
    male = v["patientSex"].lower().startswith("m")
    syms = [FR_SYMPTOM[s] for s in v["symptoms"]]
    is_child = age < 16
    third_person = is_child or any(s in OBSERVER_SYMPTOMS for s in v["symptoms"])

    if is_child:
        who = "Mon fils" if male else "Ma fille"
        return f"{who} de {age} {unit} a {_join(syms)}."
    if third_person:
        who = "un homme" if male else "une femme"
        subj = "Il" if male else "Elle"
        return f"Je vous appelle pour {who} de {age} {unit}. {subj} a {_join(syms)}."
    who = "un homme" if male else "une femme"
    return f"Je suis {who} de {age} {unit} et j'ai {_join(syms)}."


def sample_evenly(recs: list[dict], n: int) -> list[dict]:
    """Échantillon déterministe réparti (pas de graine aléatoire)."""
    if n >= len(recs):
        return recs
    step = len(recs) / n
    return [recs[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Construit le gold urgent (IyàwóBench REFER_NOW)")
    ap.add_argument("--n", type=int, default=48, help="Nombre de cas à ajouter (défaut 48)")
    ap.add_argument("--all", action="store_true", help="Utiliser les 100 REFER_NOW")
    args = ap.parse_args()

    data = json.load(open(RAW_DIR / "iyawobench" / "iyawobench_v1.json", encoding="utf-8"))
    refer_now = [v for v in data["vignettes"] if v["expected_triage"] == "REFER_NOW"]
    refer_now.sort(key=lambda v: v["vignette_id"])

    chosen = refer_now if args.all else sample_evenly(refer_now, args.n)

    out_dir = PROCESSED_DIR / "gold" / "fr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iyawobench_urgent.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for v in chosen:
            rec = make_record(
                source="iyawobench",
                case_id=v["vignette_id"],
                domain=v["disease"],
                patient_msg=build_fr_message(v),
                gold_label=v["expected_triage"],
                gold_range=IYAWOBENCH_RANGE[v["expected_triage"]],
            )
            rec["metadata"]["lang"] = "fr"
            rec["metadata"]["expected_severity"] = v["expected_severity"]
            # constantes conservées pour traçabilité, mais ABSENTES du message patient
            rec["metadata"]["vitals"] = {
                k: v[k] for k in ("temperature", "spo2", "heartRate", "systolic", "diastolic")
            }
            rec["metadata"]["original_text"] = ", ".join(v["symptoms"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{len(chosen)} cas urgents (REFER_NOW) → {out_path}")
    urgents = sum(1 for v in chosen)
    print(f"  tous en [1, 3] → binary_urgent=True ({urgents} cas)")
    print("  exemples :")
    for v in chosen[:3]:
        print(f"    {v['vignette_id']}: {build_fr_message(v)}")


if __name__ == "__main__":
    main()
