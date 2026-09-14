"""Construction de la fiche de triage (sortie structurée) et des explications.

La fiche est **déterministe** : dérivée de la règle FRENCH qui a généré le fact
sheet (niveau correct par construction). Format « tool_call simulé ».
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "Tu es un agent de triage médical pour les urgences. Tu vouvouies le patient, "
    "tu poses une question à la fois, tu es rassurant et tu n'utilises pas de jargon. "
    "Tu raisonnes en interne dans des balises <think>...</think> (faits connus, "
    "red flags, plage d'urgence, prochaine question). Quand tu as assez d'éléments, tu "
    "termines par une fiche entre [FICHE] et [/FICHE] suivie d'une explication claire au patient."
)

_EXPLANATIONS = {
    "1": "Votre situation est grave, une équipe va vous prendre en charge immédiatement, restez avec nous.",
    "2": "Votre état nécessite une prise en charge rapide, une infirmière va vous voir très vite.",
    "3a": ("Nous allons vous évaluer dans l'heure qui vient, restez en salle d'attente "
            "et prévenez-nous si ça s'aggrave."),
    "3b": "Vous allez être vu rapidement, prévenez-nous si un symptôme s'aggrave.",
    "4": "Votre situation est stable, vous serez vu dans les deux heures environ.",
    "5": "Votre situation ne semble pas urgente, vous serez vu dans la journée ; en cas d'aggravation, prévenez-nous.",
}


def build_fiche(fact: dict, rule: dict, levels: dict) -> dict:
    """Construit la fiche (tool_call) à partir du fact sheet + règle."""
    level_label = rule["level"]
    level = int(level_label[0])
    rec = levels[level_label]

    red_flags = [rule["condition"]] if rule.get("red_flag") else []
    missing = list(rule.get("objective") or [])
    for metric in (rule.get("vitals") or {}):
        if metric != "T":
            missing.append(metric)

    return {
        "name": "finalize_triage",
        "arguments": {
            "priority": level,
            "priority_range": [level, level],
            "red_flags": red_flags,
            "missing_info": missing,
            "confidence": 0.9,
            "summary": rule["condition"],
            "recommendation": f"{rec['label']} — médecin {rec['delai_medecin']}",
        },
    }


def fiche_block(fact: dict, rule: dict, levels: dict) -> str:
    """Retourne le bloc ``[FICHE] ... [/FICHE]``."""
    return "[FICHE]\n" + json.dumps(build_fiche(fact, rule, levels), ensure_ascii=False) + "\n[/FICHE]"


def explanation(rule: dict) -> str:
    """Explication naturelle au patient, selon le niveau."""
    return _EXPLANATIONS[rule["level"]]
