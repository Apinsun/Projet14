"""Générateur déterministe de fact sheets de triage (Phase 1).

À partir de la grille FRENCH encodée (``french_rules.json``), échantillonne des
fiches cliniques dont le **niveau est calculé** (correct par construction) :

1. on tire ``(catégorie, motif, règle)`` — la règle impose le ``true_level`` ;
2. on génère les constantes **à partir des plages de la règle** (cohérentes) ;
3. on dérive ``patient_observable`` (ce que le patient peut dire).

Le LLM n'intervient qu'en Phase 2 (habillage en langage naturel).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RULES_PATH = Path(__file__).parent / "french_rules.json"

# Pools (à enrichir).
_HISTORY_POOL = [
    "HTA", "diabète type 2", "tabagisme actif", "asthme", "BPCO",
    "insuffisance cardiaque", "obésité", "dyslipidémie",
    "insuffisance rénale chronique", "cardiopathie ischémique", "AVC ancien",
]

_TREATMENT_POOL = [
    "amiodarone", "metformine", "insuline", "ramipril", "bisoprolol",
    "furosémide", "aspirine", "clopidogrel", "salbutamol", "apixaban",
    "atorvastatine", "paracétamol",
]

_PEDIATRIC_HISTORY_POOL = [
    "asthme", "épilepsie", "eczéma", "allergie alimentaire", "RGO",
    "prématurité", "otites à répétition", "bronchiolites à répétition",
]

_PEDIATRIC_TREATMENT_POOL = [
    "salbutamol", "paracétamol", "valproate", "dermocorticoïdes", "budesonide",
]

# Bornes d'âge plausibles par catégorie.
_CATEGORY_AGE = {
    "cardio": (40, 85),
    "infectiologie": (1, 90),
    "abdominal": (18, 85),
    "genito_urinaire": (18, 85),
    "gyneco_obstetrique": (16, 45),
    "intoxication": (15, 70),
    "neurologie": (30, 85),
    "ophtalmologie": (5, 85),
    "orl_stomatologie": (2, 85),
    "peau": (5, 85),
    "respiratoire": (5, 85),
    "pediatrie": (1, 24),  # âge en mois
    "psychiatrie": (15, 70),
    "rhumatologie": (30, 85),
    "traumatologie": (10, 80),
    "divers": (18, 90),
}

_NORMAL_VITALS = {"PAS": 125, "FC": 75, "SpO2": 98, "FR": 16, "T": 37.0, "GCS": 15}

# Durées plausibles par niveau de triage (sans « depuis » : préfixe ajouté au rendu).
_DURATION_BY_LEVEL = {
    1: ["quelques minutes", "30 minutes", "1 heure"],
    2: ["1 heure", "quelques heures", "ce matin"],
    3: ["ce matin", "hier", "2 jours"],
    4: ["2 jours", "3 jours", "une semaine"],
    5: ["une semaine", "plusieurs jours", "2 semaines"],
}

# Vocabulaire évoquant une douleur (pour décider d'ajouter une échelle de douleur).
_PAIN_KEYWORDS = (
    "douleur", "mal", "colique", "torsion", "céphal", "brûlur", "traumat",
    "entorse", "fracture", "contusion", "lombal", "arthral", "dysurie",
)


def load_rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _sample_vitals(rule: dict, rng: random.Random) -> dict:
    """Génère des constantes cohérentes avec les plages de la règle."""
    vitals = dict(_NORMAL_VITALS)
    for metric, ranges in (rule.get("vitals") or {}).items():
        lo, hi = rng.choice(ranges)
        if metric == "T":
            vitals[metric] = round(rng.uniform(lo, hi), 1)
        else:
            vitals[metric] = rng.randint(int(lo), int(hi))
    return vitals


def _build_fact_sheet(category: str, motif: str, rule: dict, rng: random.Random, counter: dict) -> dict:
    """Construit une fact sheet à partir d'une règle donnée."""
    counter["n"] += 1
    case_id = f"{category}_{counter['n']:04d}"

    vitals = _sample_vitals(rule, rng)
    signs_objectifs = list(rule.get("objective") or [])

    is_pediatric = category == "pediatrie"
    if is_pediatric:
        history = rng.sample(_PEDIATRIC_HISTORY_POOL, k=rng.randint(0, 1))
        k_treat = rng.randint(0, 1) if history else 0
        treatments = rng.sample(_PEDIATRIC_TREATMENT_POOL, k=k_treat)
    else:
        history = rng.sample(_HISTORY_POOL, k=rng.randint(1, 2))
        k_treat = rng.randint(1, 2) if history else rng.randint(0, 1)
        treatments = rng.sample(_TREATMENT_POOL, k=k_treat)

    lo, hi = _CATEGORY_AGE[category]
    sex = "F" if category == "gyneco_obstetrique" else rng.choice(["H", "F"])
    fact = {
        "case_id": case_id,
        "category": category,
        "motif": motif,
        "age": rng.randint(lo, hi),
        "age_unit": "mois" if category == "pediatrie" else "ans",
        "sex": sex,
        "signs_objectifs": signs_objectifs,
        "vital_signs": vitals,
        "medical_history": history,
        "treatments": treatments,
        "true_level": int(rule["level"][0]),  # 3a/3b -> 3
        "level_label": rule["level"],
        "justification": f"{rule['condition']} → Tri {rule['level']}",
        "rule": rule,
    }
    fact["patient_observable"] = derive_patient_observable(fact, rule, motif, rng)
    return fact


def sample_fact_sheet(rules: dict, rng: random.Random, counter: dict) -> dict:
    """Échantillonne une fact sheet aléatoirement (catégorie, motif, règle)."""
    category = rng.choice(list(rules["categories"]))
    cat = rules["categories"][category]
    motif = rng.choice(list(cat["motifs"]))
    rule = rng.choice(cat["motifs"][motif]["rules"])
    return _build_fact_sheet(category, motif, rule, rng, counter)


def generate_balanced_fact_sheets(n_per_level: int, seed: int = 42) -> list[dict]:
    """Génère ``n_per_level`` fact sheets par niveau (1..5), équilibré."""
    rules = load_rules()
    rng = random.Random(seed)
    counter = {"n": 0}

    by_level: dict[int, list[tuple[str, str, dict]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for cname, cat in rules["categories"].items():
        for mname, m in cat["motifs"].items():
            for rule in m["rules"]:
                by_level[int(rule["level"][0])].append((cname, mname, rule))

    facts: list[dict] = []
    for lvl in (1, 2, 3, 4, 5):
        pool = by_level[lvl]
        for _ in range(n_per_level):
            cname, mname, rule = rng.choice(pool)
            facts.append(_build_fact_sheet(cname, mname, rule, rng, counter))
    rng.shuffle(facts)
    return facts


def derive_patient_observable(fact: dict, rule: dict, motif: str, rng: random.Random) -> dict:
    """Dérive la couche patient-observable à partir du fact sheet + règle.

    Enrichi : durée quasi systématique (adaptée au niveau), douleur détectée par le
    vocabulaire du motif (pas seulement le mot « douleur »).
    """
    level = int(fact["true_level"])
    can_report: dict = {
        "symptoms": [rule["symptom"]] if rule.get("symptom") else [],
        "duration": rng.choice(_DURATION_BY_LEVEL[level]),
        "pain_scale": None,
        "self_measured": {},
        "history": fact["medical_history"],
        "treatments": fact["treatments"],
    }
    cannot_report: list[str] = list(rule.get("objective") or [])

    # Constantes : la température est auto-mesurable, le reste côté infirmière.
    for metric in (rule.get("vitals") or {}):
        if metric == "T":
            can_report["self_measured"]["temperature"] = fact["vital_signs"]["T"]
        elif not any(metric in obj for obj in cannot_report):
            cannot_report.append(metric)

    # Douleur : vocabulaire évoquant la douleur (plus large que « douleur » seul).
    pain_ctx = f"{motif} {rule.get('symptom', '')} {rule.get('condition', '')}".lower()
    if any(k in pain_ctx for k in _PAIN_KEYWORDS):
        can_report["pain_scale"] = rng.randint(4, 10) if level <= 3 else rng.randint(2, 7)

    return {"can_report": can_report, "cannot_report": cannot_report}


def generate_fact_sheets(n: int, seed: int = 42) -> list[dict]:
    """Génère ``n`` fact sheets déterministes."""
    rules = load_rules()
    rng = random.Random(seed)
    counter = {"n": 0}
    return [sample_fact_sheet(rules, rng, counter) for _ in range(n)]
