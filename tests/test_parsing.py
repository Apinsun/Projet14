"""Tests du parsing de la sortie structurée (<think> / <FICHE>)."""

from __future__ import annotations

from triage_agent.parsing import (
    extract_priority,
    extract_think,
    parse_fiche,
    strip_fiche,
    strip_think,
)

RAW = (
    "<think>motif douleur, plage [1,4]</think>\n"
    '<FICHE>{"name":"finalize_triage","arguments":{"priority":4,"priority_range":[1,4]}}</FICHE>\n'
    "Depuis combien de temps avez-vous mal ?"
)


def test_extract_think():
    assert extract_think(RAW) == "motif douleur, plage [1,4]"
    assert extract_think("pas de think") is None


def test_strip_think():
    assert "<think>" not in strip_think(RAW)


def test_strip_fiche():
    assert "<FICHE>" not in strip_fiche(RAW)


def test_parse_fiche():
    fiche = parse_fiche(RAW)
    assert fiche["name"] == "finalize_triage"
    assert fiche["arguments"]["priority"] == 4


def test_parse_fiche_variant_brackets():
    assert parse_fiche('[FICHE]{"priority": 2}[/FICHE]')["priority"] == 2


def test_parse_fiche_code_fence():
    assert parse_fiche('<FICHE>```json\n{"priority": 3}\n```</FICHE>')["priority"] == 3


def test_parse_fiche_invalid():
    assert parse_fiche("<FICHE>pas du json</FICHE>") is None
    assert parse_fiche("rien") is None


def test_extract_priority():
    assert extract_priority({"arguments": {"priority": 2}}) == 2
    assert extract_priority({"priority": 3}) == 3


def test_extract_priority_prudent_bound():
    # priority absente → borne prudente = min de la plage
    assert extract_priority({"arguments": {"priority_range": [1, 4]}}) == 1


def test_extract_priority_none():
    assert extract_priority({"arguments": {}}) is None
    assert extract_priority({}) is None
