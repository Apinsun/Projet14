#!/usr/bin/env python3
"""Télécharge les datasets bruts dans ``data/raw/``.

Usage :
    poetry run python scripts/download_datasets.py            # tous
    poetry run python scripts/download_datasets.py mediqa     # un seul
"""

from __future__ import annotations

import logging
import sys

from triage_agent.data.config import DATASETS
from triage_agent.data.download import download_dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"Datasets inconnus : {unknown} (disponibles : {list(DATASETS)})")

    for name in names:
        download_dataset(name)
