"""Orchestration du pipeline de préparation des données.

Enchaîne : sélection (SFT/DPO/éval) -> anonymisation Presidio -> découpage
train/validation. Écrit les artefacts dans ``data/processed/`` :

- ``selected/``   : jeux sélectionnés (non anonymisés, pour audit) ;
- ``anonymized/`` : jeux anonymisés + ``audit.json`` (entités masquées) ;
- ``final/``      : jeux finaux ``sft_train``, ``sft_val``, ``dpo_train``,
  ``clinical_eval``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from triage_agent.data.anonymize import anonymize_records
from triage_agent.data.build import SELECTED_DIR, build_all, load_jsonl, split_sft, write_jsonl
from triage_agent.data.config import PROCESSED_DIR

logger = logging.getLogger("triage_agent.data.pipeline")

ANON_DIR = PROCESSED_DIR / "anonymized"
FINAL_DIR = PROCESSED_DIR / "final"


def _serialize_audit(audit: dict) -> dict:
    """Convertit les ``Counter`` du journal d'audit en ``dict`` sérialisables."""
    out = {
        "total": audit["total"],
        "with_pii": audit["with_pii"],
        "entities": dict(audit["entities"]),
        "detected_entities": dict(audit["detected_entities"]),
        "by_source": {},
    }
    for src, d in audit["by_source"].items():
        out["by_source"][src] = {
            "records": d["records"],
            "with_pii": d["with_pii"],
            "entities": dict(d["entities"]),
            "detected_entities": dict(d["detected_entities"]),
        }
    return out


def run_pipeline() -> dict[str, Any]:
    """Exécute tout le pipeline et retourne le manifeste final."""
    build_all()

    # 1. Anonymisation des jeux sélectionnés.
    audits: dict[str, dict] = {}
    for name in ["sft", "dpo", "clinical_eval"]:
        records = load_jsonl(SELECTED_DIR / f"{name}.jsonl")
        logger.info("Anonymisation de %s (%d enregistrements) ...", name, len(records))
        anon, audit = anonymize_records(records)
        write_jsonl(anon, ANON_DIR / f"{name}.jsonl")
        audits[name] = _serialize_audit(audit)
        logger.info("  %s : %d enregistrements avec PII détectée", name, audit["with_pii"])

    (ANON_DIR / "audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Découpage train/validation du SFT (sur les données anonymisées).
    sft_anon = load_jsonl(ANON_DIR / "sft.jsonl")
    train, val = split_sft(sft_anon)
    dpo_anon = load_jsonl(ANON_DIR / "dpo.jsonl")
    eval_anon = load_jsonl(ANON_DIR / "clinical_eval.jsonl")

    write_jsonl(train, FINAL_DIR / "sft_train.jsonl")
    write_jsonl(val, FINAL_DIR / "sft_val.jsonl")
    write_jsonl(dpo_anon, FINAL_DIR / "dpo_train.jsonl")
    write_jsonl(eval_anon, FINAL_DIR / "clinical_eval.jsonl")

    final_manifest = {
        "sft_train": len(train),
        "sft_val": len(val),
        "dpo_train": len(dpo_anon),
        "clinical_eval": len(eval_anon),
    }
    (FINAL_DIR / "manifest.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Jeux finaux : %s", final_manifest)
    return final_manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline()
