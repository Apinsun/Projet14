"""Tests de robustesse de l'API (entrées limites, erreurs, concurrence, traçabilité)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triage_agent.serving.agent import TriageAgent
from triage_agent.serving.config import Settings
from triage_agent.serving.main import app, get_agent
from triage_agent.serving.vllm_client import VLLMError

GOOD = (
    "<think>raisonnement</think>\n"
    '<FICHE>{"name":"finalize_triage","arguments":{"priority":4}}</FICHE>\n'
    "Réponse patient."
)


class _Fake:
    def chat(self, messages, **kwargs):
        return GOOD


class _Failing:
    def chat(self, messages, **kwargs):
        raise VLLMError("vLLM down")


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(audit_path=str(tmp_path / "audit.jsonl"), _env_file=None)

    def override() -> TriageAgent:
        return TriageAgent(_Fake(), settings)

    app.dependency_overrides[get_agent] = override
    yield TestClient(app), settings
    app.dependency_overrides.clear()


def test_unicode_et_speciaux(client):
    tc, _ = client
    msg = "J'ai des douleurs 😖 à l'épaule + fièvre <test> & 'guillemets' — 40°C %"
    r = tc.post("/chat", json={"messages": [{"role": "user", "content": msg}]})
    assert r.status_code == 200
    assert r.json()["parse_ok"] is True


def test_message_long(client):
    tc, _ = client
    msg = "j'ai mal au ventre " * 2000  # ~40k caractères
    r = tc.post("/chat", json={"messages": [{"role": "user", "content": msg}]})
    assert r.status_code == 200


def test_historique_vide(client):
    tc, _ = client
    r = tc.post("/chat", json={"messages": []})
    # l'app accepte (le prompt système est toujours injecté), pas de crash
    assert r.status_code == 200


def test_vllm_indisponible_502(tmp_path: Path):
    settings = Settings(audit_path=str(tmp_path / "a.jsonl"), _env_file=None)

    def override() -> TriageAgent:
        return TriageAgent(_Failing(), settings)

    app.dependency_overrides[get_agent] = override
    try:
        tc = TestClient(app)
        r = tc.post("/chat", json={"messages": [{"role": "user", "content": "bonjour"}]})
        assert r.status_code == 502
    finally:
        app.dependency_overrides.clear()


def test_concurrence_et_tracabilite(client):
    tc, settings = client

    def one(i: int) -> int:
        r = tc.post(
            "/chat",
            json={"messages": [{"role": "user", "content": f"message {i}"}], "conversation_id": f"c{i}"},
        )
        return r.status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(one, range(20)))

    assert all(c == 200 for c in codes)
    # chaque échange est tracé (une ligne = un échange), sans perte en concurrence
    lines = Path(settings.audit_path).read_text().strip().splitlines()
    assert len(lines) == 20
    # les enregistrements sont complets (champs de traçabilité obligatoires)
    required = {"_ts", "conversation_id", "messages", "raw_output", "think", "fiche", "parse_ok"}
    for line in lines:
        rec = json.loads(line)
        assert required <= set(rec), f"champs manquants : {required - set(rec)}"
