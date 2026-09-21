"""Génération de dialogues multi-tours par un LLM fort (Ollama), fiche déterministe.

Le LLM écrit la conversation naturelle : pour chaque tour de l'agent, il fournit le
raisonnement dans un champ JSON ``think`` (texte libre, sans balise) et la question ou
l'explication dans ``content``. Le code enveloppe ensuite ``think`` dans ``<think>`` et
injecte la ``<FICHE>`` **calculée par la règle FRENCH** (jamais devinée).

Pourquoi un champ séparé ? ``<think>`` est un token spécial du modèle 27B (Ollama) :
lui demander d'émettre la balise provoque des dérives (``<thought>``, ``<br/>``) ou des
arrêts prématurés. En séparant le texte du balisage, les balises sont toujours exactes.

Point clé : le patient révèle son **symptôme dès l'ouverture** (et non une ouverture
vague), pour apprendre au modèle à questionner même quand le motif est déjà annoncé.
"""

from __future__ import annotations

import json
import random
import re

from triage_agent.data.dress import _ACRONYMS, _FRENCH
from triage_agent.data.fiche import SYSTEM_PROMPT, explanation, fiche_block
from triage_agent.data.vignette import ollama_chat

MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"

# Stop tokens du modèle, sans ``<think>`` (sinon Ollama coupe dès que le modèle émet
# une balise <think>, ce qu'on lui demande explicitement).
_NO_THINK_STOP = ["<|im_start|>", "<|im_end|>", "<|im_start|>user"]

_QUESTION = {
    "durée": "Depuis combien de temps ressentez-vous cela ?",
    "intensité": "Sur une échelle de 0 à 10, à combien évaluez-vous la douleur ?",
    "température": "Avez-vous pris votre température ?",
    "antécédents": "Avez-vous des antécédents médicaux ?",
    "traitements": "Prenez-vous des médicaments ?",
}


def is_third_person_symptom(symptom: str) -> bool:
    """Vrai si le symptôme est rapporté à la 3e personne (patient inconscient / tiers)."""
    s = (symptom or "").strip().lower()
    return s.startswith(("il ", "elle ", "le patient ", "la patiente ", "il s'", "elle s'"))


_INCONSCIENT_KEYWORDS = ("respire plus", "réveille", "répond plus", "conscient", "endormi")


def is_inconscient_symptom(symptom: str) -> bool:
    """Vrai si le patient décrit est inconscient / incapable de s'exprimer (→ locuteur tiers)."""
    s = (symptom or "").strip().lower()
    return is_third_person_symptom(s) and any(k in s for k in _INCONSCIENT_KEYWORDS)

LLM_SYSTEM = (
    "Tu écris des dialogues réalistes entre un patient et un agent de triage médical aux "
    "urgences, pour entraîner un modèle de triage.\n\n"
    "Règles strictes :\n"
    "- Le patient parle en français courant, avec un ton réaliste (inquiétude proportionnée "
    "à la gravité, hésitations possibles), sans jargon médical.\n"
    "- Le patient révèle son symptôme principal dès son PREMIER message, puis UNE "
    "information à la fois, dans l'ordre indiqué.\n"
    "- Le patient ne mentionne JAMAIS de constante qu'il ne peut pas mesurer lui-même "
    "(ECG, saturation, tension artérielle, fréquence respiratoire) et ne pose JAMAIS de "
    "diagnostic.\n"
    "- L'agent vouvouie le patient, est rassurant, sans jargon, et pose UNE SEULE question "
    "à la fois.\n"
    "- Pour CHAQUE tour de l'agent, tu fournis DEUX champs :\n"
    "  * \"think\" : son raisonnement interne (motif, éventuel red flag, plage d'urgence, "
    "question suivante) ;\n"
    "  * \"content\" : sa question au patient.\n"
    "- Pour le DERNIER tour de l'agent, \"think\" contient le raisonnement final (niveau "
    "retenu, info manquante, confiance) et \"content\" contient UNIQUEMENT l'explication "
    "rassurante au patient (sans fiche, sans niveau).\n"
    "- Réponds UNIQUEMENT avec un objet JSON : "
    '{"turns": [{"role": "patient", "content": "..."}, '
    '{"role": "agent", "think": "...", "content": "..."}]}'
)


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def build_reveal_schedule(cr: dict, rng: random.Random, third_party: bool = False) -> list[tuple[str, str]]:
    """Construit la liste ordonnée (champ, info) des faits à révéler après le symptôme.

    ``third_party=True`` : un proche parle, les infos patient sont au « il ».
    """
    items: list[tuple[str, str]] = []
    if cr.get("duration"):
        items.append(("durée", f"Depuis {cr['duration']}."))
    if cr.get("pain_scale") is not None:
        items.append(("intensité", f"La douleur est à {cr['pain_scale']} sur 10."))
    if cr.get("self_measured", {}).get("temperature"):
        t = cr["self_measured"]["temperature"]
        info = f"Il a {t} °C." if third_party else f"J'ai pris ma température : {t} °C."
        items.append(("température", info))
    if cr.get("history"):
        info = f"Il a {_join(cr['history'])}." if third_party else f"J'ai {_join(cr['history'])}."
        items.append(("antécédents", info))
    if cr.get("treatments"):
        info = f"Il prend {_join(cr['treatments'])}." if third_party else f"Je prends {_join(cr['treatments'])}."
        items.append(("traitements", info))
    rng.shuffle(items)
    return items


def build_llm_user_prompt(
    fact: dict, rule: dict, levels: dict, schedule: list[tuple[str, str]], third_party: bool = False
) -> str:
    """Construit le prompt décrivant le profil du patient + les faits de triage."""
    level = int(rule["level"][0])
    unit = fact.get("age_unit", "ans")
    lines = [
        "Profil du patient :",
        f"- sexe : {fact['sex']}, âge : {fact['age']} {unit}",
        f"- Symptôme principal (à révéler dès l'ouverture) : « {rule['symptom']} »",
    ]
    if third_party:
        lines += [
            "- LOCUTEUR : un proche du patient (conjoint, parent, enfant). Le patient ne peut "
            "pas s'exprimer : le proche parle à la 1re personne pour lui-même et à la 3e personne "
            "(« il »/« elle ») pour le patient.",
        ]
    lines += [
        "",
        "Informations à révéler progressivement, DANS CET ORDRE :",
    ]
    for i, (label, info) in enumerate(schedule, 1):
        lines.append(f"{i}. {label} : « {info} »")
    lines += [
        "",
        "Tri calculé (à refléter dans ton raisonnement, sans l'écrire dans la fiche) :",
        f"- niveau : {level} ({rule['condition']})",
        "",
        "Écris le dialogue complet.",
    ]
    return "\n".join(lines)


def _parse_turns(content: str) -> list[dict] | None:
    """Parse la sortie JSON du LLM en tours ``{role, content, think?}``."""
    text = content.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(data, dict):
        return None
    turns = data.get("turns")
    if not isinstance(turns, list):
        return None
    out: list[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        role = t.get("role")
        content = t.get("content")
        if role not in ("patient", "agent") or not isinstance(content, str) or not content.strip():
            return None
        if role == "agent":
            think = t.get("think")
            if not isinstance(think, str) or not think.strip():
                return None
            out.append({"role": "agent", "content": content.strip(), "think": think.strip()})
        else:
            out.append({"role": "patient", "content": content.strip()})
    return out


def verify_turns(turns: list[dict], schedule: list[tuple[str, str]]) -> tuple[bool, str]:
    """Vérifie structure + absence de fuite + préservation des chiffres."""
    if not turns:
        return False, "vide"
    if turns[0]["role"] != "patient":
        return False, "ne commence pas par le patient"
    if turns[-1]["role"] != "agent":
        return False, "ne finit pas par l'agent"
    for i, t in enumerate(turns):
        expected = "patient" if i % 2 == 0 else "agent"
        if t["role"] != expected:
            return False, "alternance patient/agent rompue"
    patient_text = " ".join(t["content"] for t in turns if t["role"] == "patient")
    if _ACRONYMS.search(patient_text) or _FRENCH.search(patient_text):
        return False, "terme interdit côté patient"
    for _label, info in schedule:
        nums = set(re.findall(r"\d+", info))
        if nums and not nums.issubset(set(re.findall(r"\d+", patient_text))):
            return False, "chiffre manquant"
    return True, "ok"


def _final_think(rule: dict) -> str:
    """Raisonnement final déterministe (repli si le LLM dérive sur le niveau)."""
    missing = list(rule.get("objective") or [])
    for metric in (rule.get("vitals") or {}):
        if metric != "T":
            missing.append(metric)
    missing_txt = ", ".join(missing) if missing else "aucune"
    return (
        f"Niveau retenu : {rule['level']} ({rule['condition']}). "
        f"Info manquante (mesurée) : {missing_txt}. Confiance élevée."
    )


def build_dialogue_scripted(
    fact: dict, rule: dict, levels: dict, motif_label: str, schedule: list[tuple[str, str]]
) -> list[dict]:
    """Version scriptée (repli), correcte par construction, symptôme en ouverture."""
    level = int(rule["level"][0])
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    opening = rule["symptom"].strip().rstrip(".")
    messages.append({"role": "user", "content": f"Bonjour, {opening}."})

    for i, (label, info) in enumerate(schedule):
        if i == 0:
            red = "confirmé" if rule.get("red_flag") else "à préciser"
            think = (
                f"Motif : {motif_label}. Red flag ({rule['condition']}) : {red}. "
                f"Plage : [1, {level}]. Question suivante : {label}."
            )
        else:
            prev = schedule[i - 1][0]
            think = (
                f"Complément recueilli : {prev}. Plage resserrée : [{level}, {level}]. "
                f"Question suivante : {label}."
            )
        messages.append({"role": "assistant", "content": f"<think>{think}</think>\n{_QUESTION[label]}"})
        messages.append({"role": "user", "content": info})

    messages.append({
        "role": "assistant",
        "content": f"<think>{_final_think(rule)}</think>\n{fiche_block(fact, rule, levels)}\n{explanation(rule)}",
    })
    return messages


def generate_dialogue_llm(
    fact: dict,
    rule: dict,
    levels: dict,
    motif_label: str,
    model: str = MODEL,
    schedule: list[tuple[str, str]] | None = None,
    third_party: bool = False,
) -> tuple[list[dict], dict]:
    """Génère le dialogue via le LLM ; replie sur le scripté en cas d'échec."""
    if schedule is None:
        schedule = build_reveal_schedule(
            fact["patient_observable"]["can_report"], random.Random(), third_party
        )

    try:
        user = build_llm_user_prompt(fact, rule, levels, schedule, third_party)
        raw = ollama_chat(
            model, LLM_SYSTEM, user, temperature=0.7, timeout=600,
            think=False, stop=_NO_THINK_STOP,
        )
        turns = _parse_turns(raw)
        ok, reason = verify_turns(turns, schedule)
        if not ok:
            return build_dialogue_scripted(fact, rule, levels, motif_label, schedule), {
                "method": "fallback", "reason": reason,
            }

        # Assemblage : balisage <think>/<FICHE> garanti par le code.
        level = int(rule["level"][0])
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        n = len(turns)
        for i, t in enumerate(turns):
            if t["role"] == "patient":
                messages.append({"role": "user", "content": t["content"]})
                continue
            think = t["think"]
            if i == n - 1:
                # Le niveau dans le <think> final doit être correct (sinon repli).
                if str(level) not in think:
                    think = _final_think(rule)
                content = t["content"] or explanation(rule)
                messages.append({
                    "role": "assistant",
                    "content": f"<think>{think}</think>\n{fiche_block(fact, rule, levels)}\n{content}",
                })
            else:
                messages.append({"role": "assistant", "content": f"<think>{think}</think>\n{t['content']}"})

        return messages, {"method": "llm", "fiche": "programmatique"}
    except Exception as exc:  # réseau / JSON / timeout -> repli
        return build_dialogue_scripted(fact, rule, levels, motif_label, schedule), {
            "method": "fallback", "reason": f"erreur: {exc}",
        }
