#!/usr/bin/env python3
"""Pipeline complet de préparation des données : sélection -> anonymisation -> split.

Usage :
    poetry run python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging

from triage_agent.data.pipeline import run_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline()
