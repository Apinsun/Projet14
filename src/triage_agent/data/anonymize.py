"""Anonymisation RGPD des datasets via Presidio.

Stratégie (documentée dans le rapport) :

- Les 4 corpus sources sont **publics et non nominatifs** (QCM de pharmacie,
  encyclopédies de maladies, questions de patients anonymisées, Q&A synthétique).
- L'analyse Presidio (NER spaCy + recognizers regex) a montré que **toutes les
  entités détectées sont des faux positifs sur du contenu médical** :
  ``PERSON`` = éponymes de maladies (Parkinson, Guillain-Barré), ``URL`` =
  « E.coli » / « word.Word », ``MEDICAL_LICENSE`` = identifiants SNP
  (« rs1063192 ») ou d'essai clinique (« CT0252281 »), ``UK_NHS`` = numéro de
  centre antipoison, ``PHONE_NUMBER`` = valeur de dosage (« (16.40 »).
- On ne **masque donc que les identifiants financiers/contact sans ambiguïté**
  (``EMAIL_ADDRESS``, ``CREDIT_CARD``, ``IBAN_CODE``, ``IP_ADDRESS``), dont la
  regex est précise et n'a produit aucun faux positif.
- Toutes les autres entités **détectées** sont conservées dans le journal d'audit
  (``detected_entities``) comme preuve de l'analyse, sans être masquées.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger("triage_agent.data.anonymize")

# Réduit le bruit de logs de Presidio.
logging.getLogger("presidio_analyzer").setLevel(logging.ERROR)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
logging.getLogger("presidio_anonymizer").setLevel(logging.ERROR)
logging.getLogger("presidio-anonymizer").setLevel(logging.ERROR)

_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_md"},
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
    ],
}

# Identifiants sans ambiguïté que l'on masque réellement (regex précises).
PII_TYPES = {"EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"}

_DPO_FIELDS = ("prompt", "chosen", "rejected")


def _build_engine() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIG)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["en", "fr"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en", "fr"])
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer, _anonymizer = _build_engine()
    return _analyzer, _anonymizer


def anonymize_text(text: str, lang: str, score_threshold: float = 0.5) -> tuple[str, list[str], list[str]]:
    """Analyse et anonymise ``text``.

    Retourne ``(texte_anonymisé, entités_masquées, entités_détectées)``. Seules
    les entités de ``PII_TYPES`` sont masquées ; toutes les autres détections
    (``PERSON``, ``URL``…) sont retournées à titre informatif pour l'audit.
    """
    if not text:
        return text, [], []

    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(text=text, language=lang, score_threshold=score_threshold)

    detected = [r.entity_type for r in results]
    to_mask = [r for r in results if r.entity_type in PII_TYPES]

    if not to_mask:
        return text, [], detected

    anonymized = anonymizer.anonymize(text=text, analyzer_results=to_mask)
    return anonymized.text, [r.entity_type for r in to_mask], detected


def anonymize_record(record: dict) -> tuple[dict, list[str], list[str]]:
    """Anonymise un enregistrement. Retourne (copie, masquées, détectées)."""
    lang = record.get("lang", "en")
    out = dict(record)
    masked: list[str] = []
    detected: list[str] = []

    if record.get("role") == "sft":
        messages = []
        for msg in record["messages"]:
            content = msg.get("content", "")
            clean, m, d = anonymize_text(content, lang)
            messages.append({**msg, "content": clean})
            masked.extend(m)
            detected.extend(d)
        out["messages"] = messages
    elif record.get("role") == "dpo":
        for field in _DPO_FIELDS:
            clean, m, d = anonymize_text(record.get(field, ""), lang)
            out[field] = clean
            masked.extend(m)
            detected.extend(d)

    return out, masked, detected


def anonymize_records(records: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Anonymise une liste d'enregistrements et construit le journal d'audit."""
    out_records: list[dict] = []
    audit: dict[str, Any] = {
        "total": len(records),
        "with_pii": 0,
        "by_source": {},
        "entities": Counter(),  # masquées
        "detected_entities": Counter(),  # toutes détections (preuve d'analyse)
    }

    for rec in records:
        anon, masked, detected = anonymize_record(rec)
        out_records.append(anon)

        source = rec.get("source", "unknown")
        if masked:
            audit["with_pii"] += 1

        audit["by_source"].setdefault(
            source,
            {"records": 0, "with_pii": 0, "entities": Counter(), "detected_entities": Counter()},
        )
        audit["by_source"][source]["records"] += 1
        if masked:
            audit["by_source"][source]["with_pii"] += 1
        audit["entities"].update(masked)
        audit["by_source"][source]["entities"].update(masked)
        audit["detected_entities"].update(detected)
        audit["by_source"][source]["detected_entities"].update(detected)

    return out_records, audit
