#!/usr/bin/env python3
"""Prépare les jeux gold (Ramaswamy, Levine, IyàwóBench) au format d'évaluation.

Usage :
    poetry run python scripts/prepare_gold_eval.py [--include-iyawobench]
"""

from __future__ import annotations

import argparse
import csv
import json

from triage_agent.data.config import PROCESSED_DIR, RAW_DIR
from triage_agent.eval.gold import (
    IYAWOBENCH_RANGE,
    LEVINE_RANGE,
    RAMASWAMY_RANGE,
    make_record,
)


def load_ramaswamy() -> list[dict]:
    rows = list(csv.DictReader(open(RAW_DIR / "ramaswamy" / "vignettes.csv", encoding="utf-8")))
    records = []
    for r in rows:
        if r["prompt_type"] != "0":  # 0 = symptômes seuls (patient-reportable)
            continue
        label = r["gold_triage"].strip()
        records.append(make_record(
            source="ramaswamy",
            case_id=r["case_id"],
            domain=r["domain"],
            patient_msg=r["input_prompt"],
            gold_label=label,
            gold_range=RAMASWAMY_RANGE[label],
        ))
    return records


def load_levine() -> list[dict]:
    rows = list(csv.DictReader(
        open(RAW_DIR / "levine" / "vignettes-2020.tsv", encoding="utf-8"), delimiter="\t"
    ))
    records = []
    for r in rows:
        label = r["Correct Triage"].strip()
        patient = f"{r['Current Problem'].strip()}. {r['Additional Details'].strip()}"
        records.append(make_record(
            source="levine",
            case_id=r["Case #"],
            domain="general",
            patient_msg=patient,
            gold_label=label,
            gold_range=LEVINE_RANGE[label],
        ))
    return records


def load_iyawobench() -> list[dict]:
    data = json.load(open(RAW_DIR / "iyawobench" / "iyawobench_v1.json", encoding="utf-8"))
    records = []
    for v in data["vignettes"]:
        label = v["expected_triage"]
        age = f"{v['patientAge']} {v['patientAgeUnit']}"
        sex = "female" if v["patientSex"].lower().startswith("f") else "male"
        symptoms = ", ".join(s["lower"] if isinstance(s, str) else str(s) for s in v["symptoms"])
        patient = f"I am {age} old, {sex}. My symptoms: {symptoms}."
        records.append(make_record(
            source="iyawobench",
            case_id=v["vignette_id"],
            domain=v["disease"],
            patient_msg=patient,
            gold_label=label,
            gold_range=IYAWOBENCH_RANGE[label],
        ))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Prépare les jeux gold au format d'évaluation")
    ap.add_argument("--include-iyawobench", action="store_true",
                    help="Inclure IyàwóBench (constantes non-observables → optionnel)")
    args = ap.parse_args()

    out_dir = PROCESSED_DIR / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = [("ramaswamy", load_ramaswamy), ("levine", load_levine)]
    if args.include_iyawobench:
        loaders.append(("iyawobench", load_iyawobench))

    for name, loader in loaders:
        records = loader()
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels = {}
        for rec in records:
            lab = rec["metadata"]["gold_label_original"]
            labels[lab] = labels.get(lab, 0) + 1
        print(f"{name}: {len(records)} cas → {path}")
        print(f"   labels : {labels}")


if __name__ == "__main__":
    main()
