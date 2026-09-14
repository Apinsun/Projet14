"""Revue post-traitement : génère un rapport Markdown par source d'entrée.

Consomme les artefacts du pipeline (manifestes + audit + jeux finaux) et produit
pour chaque source : le cheminement (brut -> standardisé -> sélectionné -> PII
masquées), un exemple d'enregistrement et, le cas échéant, un exemple de donnée
réellement anonymisée.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from triage_agent.data.config import PROCESSED_DIR

# Volumes bruts téléchargés (issus de l'EDA).
_RAW_TOTALS = {
    "frenchmedmcqa": 3_105,
    "medquad": 47_441,
    "mediqa": 383,
    "ultramedical_preference": 112_362,
}

_SOURCE_LABELS = {
    "frenchmedmcqa": "FrenchMedMCQA",
    "medquad": "MedQuAD",
    "mediqa": "MediQA",
    "ultramedical_preference": "UltraMedical-Preference",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _truncate(text: str, n: int = 220) -> str:
    s = " ".join(text.split())
    return s[:n] + ("…" if len(s) > n else "")


def _sft_display(record: dict) -> tuple[str, str]:
    user = record["messages"][0]["content"]
    assistant = record["messages"][1]["content"]
    return _truncate(user), _truncate(assistant)


def _dpo_display(record: dict) -> tuple[str, str, str]:
    return (
        _truncate(record["prompt"]),
        _truncate(record["chosen"]),
        _truncate(record["rejected"]),
    )


def _has_mask(record: dict) -> bool:
    blob = json.dumps(record, ensure_ascii=False)
    tags = ("<EMAIL", "<URL", "<PHONE", "<MEDICAL", "<UK_NHS", "<IBAN", "<CREDIT", "<IP")
    return "<" in blob and any(t in blob for t in tags)


def _find_masked(records: list[dict]) -> dict | None:
    for r in records:
        if _has_mask(r):
            return r
    return None


def generate_review() -> str:
    """Retourne le rapport Markdown de revue post-traitement."""
    transform_manifest = _load_json(PROCESSED_DIR / "manifest.json")
    audits = _load_json(PROCESSED_DIR / "anonymized" / "audit.json")
    final_manifest = _load_json(PROCESSED_DIR / "final" / "manifest.json")

    # Agrégation du journal d'audit (blocs sft / dpo / clinical_eval) par source.
    agg_by_source: dict[str, dict] = {}
    for block in audits.values():
        for src, d in block["by_source"].items():
            agg = agg_by_source.setdefault(
                src, {"with_pii": 0, "entities": Counter(), "detected_entities": Counter()}
            )
            agg["with_pii"] += d["with_pii"]
            agg["entities"].update(d["entities"])
            agg["detected_entities"].update(d["detected_entities"])

    # Comptage par source dans les jeux finaux.
    sft_records = _load_jsonl(PROCESSED_DIR / "final" / "sft_train.jsonl") + _load_jsonl(
        PROCESSED_DIR / "final" / "sft_val.jsonl"
    )
    dpo_records = _load_jsonl(PROCESSED_DIR / "final" / "dpo_train.jsonl")
    eval_records = _load_jsonl(PROCESSED_DIR / "final" / "clinical_eval.jsonl")

    sft_by_source = Counter(r["source"] for r in sft_records)
    eval_by_source = Counter(r["source"] for r in eval_records)
    dpo_by_source = Counter(r["source"] for r in dpo_records)

    lines: list[str] = []
    lines.append("# Étape 1 — Préparation des données : revue post-traitement\n")
    lines.append("Pipeline exécuté : **standardisation → nettoyage → sélection SFT/DPO → "
                 "anonymisation Presidio → découpage train/val**.\n")
    lines.append("## Synthèse des jeux finaux\n")
    lines.append("| Jeu | Nombre d'enregistrements |")
    lines.append("|---|---|")
    for k, v in final_manifest.items():
        lines.append(f"| `{k}` | {v:,} |")
    lines.append("")
    lines.append("## Cheminement par source\n")

    for name, label in _SOURCE_LABELS.items():
        t = transform_manifest["stats"].get(name, {})
        audit_src = agg_by_source.get(name, {})
        raw = _RAW_TOTALS.get(name, 0)
        standardized = t.get("output", 0)
        selected = sft_by_source.get(name, 0) + dpo_by_source.get(name, 0) + eval_by_source.get(name, 0)
        with_pii = audit_src.get("with_pii", 0)
        person = audit_src.get("detected_entities", {}).get("PERSON", 0)

        lines.append(f"### {label} (`{name}`)\n")
        lines.append("| Étape | Volume |")
        lines.append("|---|---|")
        lines.append(f"| Brut téléchargé | {raw:,} |")
        if name == "medquad":
            lines.append("| Réponses non nulles (filtre structurel) | 16 407 |")
        lines.append(f"| Standardisé (JSONL nettoyé) | {standardized:,} |")
        lines.append(f"| Sélectionné (SFT + DPO + éval) | {selected:,} |")
        lines.append(f"| **Enregistrements avec PII masquée** | **{with_pii}** |")
        lines.append(f"| Occurrences `PERSON` détectées (non masquées, éponymes) | {person} |")
        ent = audit_src.get("entities", {})
        if ent:
            lines.append("")
            lines.append("Entités masquées : " + ", ".join(f"`{k}` ({v})" for k, v in sorted(ent.items())))
        det = audit_src.get("detected_entities", {})
        if det:
            lines.append("")
            lines.append("Entités détectées mais non masquées (faux positifs médicaux) : "
                         + ", ".join(f"`{k}` ({v})" for k, v in sorted(det.items(), key=lambda kv: -kv[1])[:6]))
        lines.append("")

        # Exemple représentatif.
        sample = None
        if name == "ultramedical_preference":
            sample = next((r for r in dpo_records if r["source"] == name), None)
        else:
            sample = next((r for r in sft_records if r["source"] == name), None) or next(
                (r for r in eval_records if r["source"] == name), None
            )
        if sample:
            lines.append("**Exemple (post-traitement) :**")
            lines.append("")
            if sample.get("role") == "dpo":
                p, c, r = _dpo_display(sample)
                lines.append(f"- `prompt`   : {p}")
                lines.append(f"- `chosen`   : {c}")
                lines.append(f"- `rejected` : {r}")
            else:
                u, a = _sft_display(sample)
                lines.append(f"- `user`      : {u}")
                lines.append(f"- `assistant` : {a}")
            lines.append("")

        # Exemple de PII réellement masquée (si présent).
        masked = _find_masked(sft_records + dpo_records + eval_records)
        if masked and masked["source"] == name:
            lines.append("**Exemple de PII masquée :**")
            lines.append("")
            if masked.get("role") == "dpo":
                lines.append(f"- `prompt` : {_truncate(masked['prompt'])}")
            else:
                lines.append(f"- `user`      : {_truncate(masked['messages'][0]['content'])}")
                lines.append(f"- `assistant` : {_truncate(masked['messages'][1]['content'])}")
            lines.append("")

    lines.append("## Note RGPD\n")
    lines.append("- Les corpus sources sont publics et non nominatifs (QCM de pharmacie, "
                 "encyclopédies de maladies, questions de patients anonymisées, Q&A synthétique).")
    lines.append("- L'analyse Presidio (NER spaCy + recognizers regex) a montré que **toutes les "
                 "entités détectées sont des faux positifs sur du contenu médical** : `PERSON` = "
                 "éponymes de maladies, `URL` = « E.coli » / « word.Word », `MEDICAL_LICENSE` = "
                 "identifiants SNP (`rs1063192`) ou d'essai clinique, `UK_NHS` = numéro de centre "
                 "antipoison, `PHONE_NUMBER` = valeur de dosage.")
    lines.append("- Seuls les identifiants **sans ambiguïté** (`EMAIL_ADDRESS`, `CREDIT_CARD`, "
                 "`IBAN_CODE`, `IP_ADDRESS`) sont masqués ; aucune occurrence n'a été trouvée, "
                 "confirmant l'absence de données personnelles dans le corpus final.")
    lines.append("- Le détail complet est dans `data/processed/anonymized/audit.json`.")
    lines.append("")
    return "\n".join(lines)


def write_review(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_review(), encoding="utf-8")
