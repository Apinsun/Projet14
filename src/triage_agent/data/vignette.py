"""Génération de vignettes patient via un LLM local (Ollama) — Phase 2.

Le LLM « habille » un fact sheet en message de patient naturel, **sans changer
les faits** ni révéler ce que le patient ne peut pas savoir (constantes mesurées,
ECG…). Il ne décide jamais du niveau de triage.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger("triage_agent.data.vignette")

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "Leila_fast:latest"


def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    timeout: int = 180,
) -> str:
    """Appelle l'API Ollama (/api/chat) et retourne le contenu de la réponse."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def build_user_prompt(fact: dict) -> str:
    """Construit le prompt décrivant ce que le patient connaît et peut dire."""
    po = fact["patient_observable"]
    cr = po["can_report"]
    lines = [f"Patient : sexe {fact['sex']}, {fact['age']} ans.", ""]
    lines.append("Informations que le patient connaît et peut exprimer :")
    if cr["symptoms"]:
        lines.append("- symptômes ressentis : " + "; ".join(cr["symptoms"]))
    if cr["pain_scale"] is not None:
        lines.append(f"- intensité de la douleur : {cr['pain_scale']}/10")
    if cr["duration"]:
        lines.append(f"- depuis : {cr['duration']}")
    if cr["self_measured"].get("temperature"):
        lines.append(f"- température mesurée au thermomètre : {cr['self_measured']['temperature']} °C")
    if cr["history"]:
        lines.append("- antécédents connus : " + ", ".join(cr["history"]))
    if cr["treatments"]:
        lines.append("- traitements en cours : " + ", ".join(cr["treatments"]))
    lines.append("")
    lines.append("Rédige le message d'ouverture du patient.")
    return "\n".join(lines)


def generate_vignette(fact: dict, system: str, model: str = DEFAULT_MODEL) -> str:
    """Génère la vignette (message patient) pour un fact sheet."""
    user = build_user_prompt(fact)
    return ollama_chat(model, system, user)


def parse_json_vignette(content: str) -> str:
    """Extrait le champ ``vignette`` si le modèle répond en JSON, sinon renvoie le brut."""
    text = content.strip()
    # Retire les éventuelles fences markdown (```json ... ```).
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    if text.startswith("{"):
        try:
            data: Any = json.loads(text)
            if isinstance(data, dict) and "vignette" in data:
                return data["vignette"]
        except json.JSONDecodeError:
            pass
    return content
