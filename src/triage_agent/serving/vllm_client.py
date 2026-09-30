"""Client vLLM (API OpenAI-compatible), via httpx."""

from __future__ import annotations

import httpx


class VLLMError(RuntimeError):
    """Erreur de communication avec vLLM."""


class VLLMClient:
    """Client synchrone vers ``/v1/chat/completions`` de vLLM."""

    def __init__(self, base_url: str, model: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> str:
        """Appelle vLLM et retourne le ``content`` généré."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise VLLMError(f"vLLM injoignable ou en erreur : {exc}") from exc
