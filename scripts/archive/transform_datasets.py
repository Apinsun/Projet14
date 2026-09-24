#!/usr/bin/env python3
"""Nettoyage et standardisation des datasets bruts vers ``data/processed/``.

Usage :
    poetry run python scripts/transform_datasets.py
"""

from __future__ import annotations

import logging

from triage_agent.data.transform import transform_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transform_all()
