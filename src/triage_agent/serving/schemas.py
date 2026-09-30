"""Schémas Pydantic de l'API (requête / réponse).

La fiche de triage produite par le modèle est renvoyée **telle quelle** (dict), sans
validation stricte : le modèle ajoute parfois des champs libres (``description``,
``severity``…). On ne la re-calcule jamais — c'est l'objet du POC de la mesurer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    raw: str  # sortie brute complète (à réinjecter dans l'historique)
    reply: str  # texte patient : sans <think> ni <FICHE>
    think: str | None = None  # raisonnement interne (affiché « thinking »)
    fiche: dict[str, Any] | None = None  # fiche parsée (None si malformée)
    parse_ok: bool
    conversation_id: str | None = None
