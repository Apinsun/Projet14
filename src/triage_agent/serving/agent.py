"""Orchestration d'un tour : prompt → vLLM → parse → log d'audit."""

from __future__ import annotations

from triage_agent.parsing import extract_think, parse_fiche, strip_fiche, strip_think
from triage_agent.serving.audit import log_exchange
from triage_agent.serving.config import SYSTEM_PROMPT, Settings


class TriageAgent:
    """Agent de triage : appelle le modèle et structure la réponse pour le client.

    ``client`` est duck-typé (tout objet exposant ``chat(messages, **kwargs) -> str``),
    ce qui permet de le remplacer par un mock dans les tests (pas de GPU requis).
    """

    def __init__(self, client, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def handle(self, messages: list[dict], conversation_id: str | None = None) -> dict:
        full = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        raw = self.client.chat(
            full,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            max_tokens=self.settings.max_tokens,
        )

        think = extract_think(raw)
        fiche = parse_fiche(raw)
        result = {
            "raw": raw,
            "reply": strip_fiche(strip_think(raw)),
            "think": think,
            "fiche": fiche,
            "parse_ok": fiche is not None,
            "conversation_id": conversation_id,
        }

        log_exchange(
            self.settings.audit_path,
            {
                "conversation_id": conversation_id,
                "messages": messages,
                "raw_output": raw,
                "think": think,
                "fiche": fiche,
                "parse_ok": result["parse_ok"],
            },
        )
        return result
