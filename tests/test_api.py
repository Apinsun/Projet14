"""Tests d'intégration de l'API FastAPI — vLLM mocké (pas de GPU requis)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triage_agent.serving.agent import TriageAgent
from triage_agent.serving.config import Settings
from triage_agent.serving.main import app, get_agent

GOOD = (
    "<think>raisonnement interne de test</think>\n"
    '<FICHE>{"name":"finalize_triage","arguments":{"priority":4,"priority_range":[4,4],"confidence":0.9}}</FICHE>\n'
    "Votre situation est stable, un médecin vous verra dans les deux heures."
)

BAD = "<think>raisonnement</think>\nVotre situation est stable, pas de fiche."


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.text


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(audit_path=str(tmp_path / "audit.jsonl"), _env_file=None)
    fake = _FakeClient(GOOD)

    def override() -> TriageAgent:
        return TriageAgent(fake, settings)

    app.dependency_overrides[get_agent] = override
    yield TestClient(app), fake, settings
    app.dependency_overrides.clear()


def test_chat_ok(client):
    tc, fake, settings = client
    resp = tc.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "bonjour, j'ai mal au ventre"}],
            "conversation_id": "c1",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["parse_ok"] is True
    assert data["think"] == "raisonnement interne de test"
    assert data["fiche"]["arguments"]["priority"] == 4
    assert data["reply"] == "Votre situation est stable, un médecin vous verra dans les deux heures."
    assert "<think>" not in data["reply"] and "<FICHE>" not in data["reply"]

    # le prompt système est injecté en tête, l'historique est passé tel quel
    assert fake.calls[0][0]["role"] == "system"
    assert fake.calls[0][1] == {"role": "user", "content": "bonjour, j'ai mal au ventre"}

    # audit écrit (une ligne = un échange)
    lines = Path(settings.audit_path).read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["conversation_id"] == "c1"
    assert rec["parse_ok"] is True


def test_chat_malformed_fiche(tmp_path: Path):
    settings = Settings(audit_path=str(tmp_path / "audit.jsonl"), _env_file=None)
    fake = _FakeClient(BAD)

    def override() -> TriageAgent:
        return TriageAgent(fake, settings)

    app.dependency_overrides[get_agent] = override
    try:
        tc = TestClient(app)
        resp = tc.post("/chat", json={"messages": [{"role": "user", "content": "bonjour"}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["parse_ok"] is False
        assert data["fiche"] is None
        assert "Votre situation est stable" in data["reply"]
    finally:
        app.dependency_overrides.clear()


def test_health():
    tc = TestClient(app)
    assert tc.get("/health").json()["status"] == "ok"


def test_index_served():
    tc = TestClient(app)
    resp = tc.get("/")
    assert resp.status_code == 200
    assert "triage" in resp.text.lower()
