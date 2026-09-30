"""Utilitaires d'extraction de la sortie structurée du modèle.

Centralise le parsing des balises ``<think>`` et ``<FICHE>``/``[FICHE]``, partagé
entre l'évaluation (``eval/harness.py``) et le serving (``serving/``).
"""

from __future__ import annotations

import json
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_FICHE_RE = re.compile(r"(?:\[FICHE\]|<FICHE>)\s*(.*?)\s*(?:\[/FICHE\]|</FICHE>)", re.S)


def extract_think(text: str) -> str | None:
    """Retourne le contenu de ``<think>…</think>`` (sans les balises), ou ``None``."""
    m = _THINK_RE.search(text or "")
    return m.group(1).strip() if m else None


def strip_think(text: str) -> str:
    """Retire le bloc ``<think>…</think>`` de la sortie."""
    return _THINK_RE.sub("", text or "").strip()


def strip_fiche(text: str) -> str:
    """Retire le bloc fiche ``[FICHE]``/``<FICHE>`` de la sortie (texte patient seul)."""
    return _FICHE_RE.sub("", text or "").strip()


def parse_fiche(text: str) -> dict | None:
    """Extrait et parse le JSON entre ``[FICHE]``/``<FICHE>``. ``None`` si absent/invalide."""
    m = _FICHE_RE.search(text or "")
    if not m:
        return None
    raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", m.group(1)).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_priority(fiche: dict) -> int | None:
    """Lit la priorité de la fiche (``arguments.priority``, sinon borne prudente de la plage)."""
    try:
        args = fiche.get("arguments", fiche)
        prio = args.get("priority")
        if prio is None:
            rng = args.get("priority_range")
            if rng:
                prio = min(int(x) for x in rng)  # borne prudente = la plus urgente
        return int(prio) if prio is not None else None
    except (TypeError, ValueError, AttributeError):
        return None
