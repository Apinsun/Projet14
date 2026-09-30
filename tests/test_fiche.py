"""Tests de la construction déterministe de la fiche (data/fiche.py)."""

from __future__ import annotations

from triage_agent.data.fiche import build_fiche, explanation


def test_build_fiche_level4():
    rule = {
        "level": "4",
        "condition": "douleur abdominale isolée",
        "red_flag": False,
        "objective": [],
        "vitals": None,
    }
    levels = {"4": {"label": "Atteinte fonctionnelle/lésionnelle stable, acte limité", "delai_medecin": "< 120 min"}}
    fiche = build_fiche({}, rule, levels)
    assert fiche["name"] == "finalize_triage"
    assert fiche["arguments"]["priority"] == 4
    assert fiche["arguments"]["priority_range"] == [4, 4]
    assert fiche["arguments"]["red_flags"] == []


def test_build_fiche_red_flag():
    rule = {"level": "1", "condition": "arrêt cardiorespiratoire", "red_flag": True, "objective": [], "vitals": None}
    levels = {"1": {"label": "Détresse vitale majeure", "delai_medecin": "immédiat"}}
    fiche = build_fiche({}, rule, levels)
    assert fiche["arguments"]["priority"] == 1
    assert fiche["arguments"]["red_flags"] == ["arrêt cardiorespiratoire"]


def test_build_fiche_vitals_missing():
    rule = {
        "level": "2",
        "condition": "hypotension",
        "red_flag": True,
        "objective": [],
        "vitals": {"PAS": [[70, 90]], "FC": [[100, 200]]},
    }
    levels = {"2": {"label": "Atteinte patente d'un organe", "delai_medecin": "< 20 min"}}
    fiche = build_fiche({}, rule, levels)
    assert "PAS" in fiche["arguments"]["missing_info"]
    assert "FC" in fiche["arguments"]["missing_info"]


def test_explanation():
    assert "immédiatement" in explanation({"level": "1"})
    assert "stable" in explanation({"level": "4"})
