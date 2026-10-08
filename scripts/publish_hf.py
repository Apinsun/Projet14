#!/usr/bin/env python3
"""Publication du dataset de triage sur HuggingFace Hub.

Publie sous `apinsun/triage-french` :
  - data/sft_vignettes.jsonl             (SFT mono-tour, 300)
  - data/sft_multiturn_fiche_nat.jsonl   (SFT multi-tours, 1 074)
  - data/dpo_quality_v3.jsonl            (DPO, 300 paires)
  - README.md (datacard = scripts/hf_dataset_card.md)

Le token est lu depuis `.env` (HF_TOKEN_WRITE, sinon HF_TOKEN) via python-dotenv.

Usage :
    poetry run python scripts/publish_hf.py [--repo apinsun/triage-french] [--private]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "triage"
CARD = Path(__file__).resolve().parent / "hf_dataset_card.md"

FILES = [
    ("data/sft_vignettes.jsonl", DATA / "sft_vignettes.jsonl"),
    ("data/sft_multiturn_fiche_nat.jsonl", DATA / "sft_multiturn_fiche_nat.jsonl"),
    ("data/dpo_quality_v3.jsonl", DATA / "dpo_quality_v3.jsonl"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Publie le dataset de triage sur HF Hub")
    ap.add_argument("--repo", default="apinsun/triage-french")
    ap.add_argument("--private", action="store_true", help="créer le dataset en privé")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN_WRITE") or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN_WRITE (ou HF_TOKEN) absent — voir .env.", file=sys.stderr)
        return 1

    for _, local in FILES:
        if not local.exists():
            print(f"❌ fichier manquant : {local}", file=sys.stderr)
            return 1

    api = HfApi(token=token)
    try:
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    except Exception as exc:
        print(f"❌ création du repo impossible ({exc}). Le token a-t-il les droits d'écriture ?", file=sys.stderr)
        return 1

    for path_in_repo, local in FILES:
        print(f"↑ {local.name} → {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type="dataset",
        )

    print("↑ datacard → README.md")
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )

    print(f"✅ publié : https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
