#!/usr/bin/env python3
"""Génère le rapport de revue post-traitement par source.

Usage :
    poetry run python scripts/run_review.py
"""

from __future__ import annotations

from triage_agent.data.config import REPO_ROOT
from triage_agent.data.review import write_review

if __name__ == "__main__":
    out = REPO_ROOT / "reports" / "02_preparation_donnees.md"
    write_review(out)
    print(f"Rapport écrit dans {out}")
