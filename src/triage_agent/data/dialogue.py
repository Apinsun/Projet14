"""Génération de dialogues de triage (multi-tours, scriptés, déterministes).

Le patient ne révèle les faits **qu'au fil des questions** de l'agent :

1. ouverture vague du patient ;
2. l'agent identifie le motif → le patient donne son symptôme ;
3. l'agent enchaîne les questions (durée, intensité, température, antécédents,
   traitements) avec un ``<think>`` qui resserre la plage d'urgence ;
4. l'agent conclut par ``<think>`` + fiche + explication.

Le niveau final est **déterministe** (règle FRENCH). Scripté pour garantir la
correction ; un habillage LLM pourra suivre.
"""

from __future__ import annotations

from triage_agent.data.fiche import SYSTEM_PROMPT, explanation, fiche_block


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def _patient_answers(cr: dict) -> list[tuple[str, str, str]]:
    """Construit la liste (champ, question, réponse) des faits à révéler."""
    qa: list[tuple[str, str, str]] = []
    if cr.get("duration"):
        qa.append(("durée", "Depuis combien de temps ressentez-vous cela ?", f"Depuis {cr['duration']}."))
    if cr.get("pain_scale") is not None:
        qa.append(
            ("intensité", "Sur une échelle de 0 à 10, à combien évaluez-vous la douleur ?",
             f"Je dirais {cr['pain_scale']} sur 10.")
        )
    if cr.get("self_measured", {}).get("temperature"):
        qa.append(
            ("température", "Avez-vous pris votre température ?",
             f"Oui, j'ai {cr['self_measured']['temperature']} °C.")
        )
    if cr.get("history"):
        qa.append(("antécédents", "Avez-vous des antécédents médicaux ?", f"J'ai {_join(cr['history'])}."))
    if cr.get("treatments"):
        qa.append(("traitements", "Prenez-vous des médicaments ?", f"Je prends {_join(cr['treatments'])}."))
    return qa


def build_dialogue(fact: dict, rule: dict, levels: dict, motif_label: str) -> list[dict]:
    """Construit un dialogue SFT multi-tours (liste de messages)."""
    level = int(rule["level"][0])
    cr = fact["patient_observable"]["can_report"]
    qa = _patient_answers(cr)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 1. Ouverture vague + identification du motif.
    messages.append({"role": "user", "content": "Bonjour, je ne me sens pas bien, j'aurais besoin d'être vu."})
    messages.append({
        "role": "assistant",
        "content": "<think>Accueil. Objectif : identifier le motif de recours.</think>\n"
                   "Bonjour, je vous écoute. Qu'est-ce qui vous amène ?",
    })
    messages.append({"role": "user", "content": rule["symptom"]})

    # 2. Questionnement progressif.
    for i, (field, question, answer) in enumerate(qa):
        if i == 0:
            red = "confirmé" if rule.get("red_flag") else "à préciser"
            think = (
                f"<think>Motif : {motif_label}. Red flag ({rule['condition']}) : {red}. "
                f"Plage : [1, {level}]. Question suivante : {field}.</think>"
            )
        else:
            prev_field = qa[i - 1][0]
            think = (
                f"<think>Complément recueilli : {prev_field}. Plage resserrée : [{level}, {level}]. "
                f"Question suivante : {field}.</think>"
            )
        messages.append({"role": "assistant", "content": f"{think}\n{question}"})
        messages.append({"role": "user", "content": answer})

    # 3. Conclusion.
    missing = list(rule.get("objective") or [])
    for metric in (rule.get("vitals") or {}):
        if metric != "T":
            missing.append(metric)
    missing_txt = ", ".join(missing) if missing else "aucune"
    think = (
        f"<think>Niveau retenu : {rule['level']} ({rule['condition']}). "
        f"Info manquante (mesurée) : {missing_txt}. Confiance élevée.</think>"
    )
    final_content = f"{think}\n{fiche_block(fact, rule, levels)}\n{explanation(rule)}"
    messages.append({"role": "assistant", "content": final_content})

    return messages
