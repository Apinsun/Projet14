"""Tests du harness d'évaluation (comparaison priorité / plage gold)."""

from __future__ import annotations

from triage_agent.eval.harness import correct, extract_priority, parse_fiche


def test_correct_in_range():
    assert correct(3, [2, 4]) == {"correct": True, "under": 0, "over": 0}


def test_correct_under():
    res = correct(5, [2, 4])
    assert res["correct"] is False
    assert res["under"] == 1
    assert res["over"] == 0


def test_correct_over():
    res = correct(1, [2, 4])
    assert res["correct"] is False
    assert res["over"] == 1
    assert res["under"] == 0


def test_harness_reexports_parsing():
    # le refactor ne casse pas les importations depuis harness
    assert parse_fiche('[FICHE]{"priority": 2}[/FICHE]')["priority"] == 2
    assert extract_priority({"priority_range": [1, 4]}) == 1
