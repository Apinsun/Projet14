"""Tests du journal d'audit (append-only JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from triage_agent.serving.audit import log_exchange


def test_log_exchange_append(tmp_path: Path):
    p = tmp_path / "sub" / "audit.jsonl"
    log_exchange(p, {"a": 1, "message": "bonjour"})
    log_exchange(p, {"a": 2, "message": "merci"})

    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["a"] == 1
    assert r0["message"] == "bonjour"
    assert "_ts" in r0  # horodatage automatique
