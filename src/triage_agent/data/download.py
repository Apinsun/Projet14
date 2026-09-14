"""Téléchargement des datasets depuis le Hub Hugging Face.

Chaque corpus est chargé puis sauvegardé en Parquet (un fichier par split) dans
``data/raw/<nom>/``.

Cas particulier : ``qanastek/frenchmedmcqa`` repose sur un script de chargement
Python, désormais non supporté par ``datasets`` (``trust_remote_code`` a été
retiré). On télécharge donc directement l'archive source ``DEFT-2023-FULL.zip``
hébergée sur le Hub et on la parse nous-mêmes (voir :func:`_download_frenchmedmcqa`).

Le token Hugging Face est lu depuis ``.env`` (variable ``HF_TOKEN``), récupéré
automatiquement par ``datasets`` / ``huggingface_hub``.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from triage_agent.data.config import DATASETS, RAW_DIR

logger = logging.getLogger("triage_agent.data.download")

_FRENCHMEDMCQA_ZIP = "DEFT-2023-FULL.zip"
# Ordre des splits du zip source -> nom de fichier JSON interne.
_FRENCHMEDMCQA_SPLITS = {"train": "train.json", "validation": "dev.json", "test": "test.json"}


def load_hf_token() -> str | None:
    """Charge ``HF_TOKEN`` depuis ``.env`` s'il n'est pas déjà dans l'environnement."""
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    return os.environ.get("HF_TOKEN")


def _download_frenchmedmcqa(out_dir: Path) -> dict[str, Path]:
    """Télécharge et parse FrenchMedMCQA depuis l'archive source DEFT-2023."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()

    zip_path = hf_hub_download(
        repo_id=DATASETS["frenchmedmcqa"]["hf_id"],
        filename=_FRENCHMEDMCQA_ZIP,
        repo_type="dataset",
        token=token,
        local_dir=out_dir,
    )
    logger.info("Archive téléchargée : %s", zip_path)

    paths: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for split, inner in _FRENCHMEDMCQA_SPLITS.items():
            with zf.open(inner) as fh:
                raw_rows = json.load(fh)

            rows = []
            for d in raw_rows:
                ans = d["answers"]
                rows.append(
                    {
                        "id": d["id"],
                        "question": d["question"],
                        "answer_a": ans["a"],
                        "answer_b": ans["b"],
                        "answer_c": ans["c"],
                        "answer_d": ans["d"],
                        "answer_e": ans["e"],
                        "correct_answers": d["correct_answers"],
                        "subject_name": d.get("subject_name"),
                        "type": d.get("type", "simple" if d["nbr_correct_answers"] == 1 else "multiple"),
                        "nbr_correct_answers": int(d["nbr_correct_answers"]),
                    }
                )

            dest = out_dir / f"{split}.parquet"
            Dataset.from_list(rows).to_parquet(dest)
            paths[split] = dest
            logger.info("  %s -> %s (%d lignes)", split, dest, len(rows))

    return paths


def download_parquet_config(hf_id: str, config: str, out_dir: Path) -> dict[str, Path]:
    """Télécharge un config spécifique d'un dataset via le datasets-server.

    Récupère les fichiers Parquet déjà convertis pour un ``config`` donné et les
    enregistre dans ``out_dir/<split>.parquet``.
    """
    import requests

    token = load_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    url = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for f in resp.json().get("parquet_files", []):
        if f["config"] != config:
            continue
        split = f["split"]
        data = requests.get(f["url"], headers=headers, timeout=120)
        data.raise_for_status()
        dest = out_dir / f"{split}.parquet"
        dest.write_bytes(data.content)
        paths[split] = dest
        logger.info("  [%s] %s -> %s (%d octets)", config, split, dest, len(data.content))
    return paths


def _download_via_parquet_server(name: str, out_dir: Path) -> dict[str, Path]:
    """Télécharge un dataset via les parquets convertis du datasets-server.

    Utile pour les datasets reposant sur un script de chargement Python, non
    supporté par ``datasets>=4``.
    """
    spec = DATASETS[name]
    return download_parquet_config(spec["hf_id"], spec["config"], out_dir)


def download_dataset(name: str) -> dict[str, Path]:
    """Télécharge un dataset et retourne ``{split: chemin_parquet}``."""
    from datasets import DatasetDict, load_dataset

    spec = DATASETS[name]
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if name == "frenchmedmcqa":
        return _download_frenchmedmcqa(out_dir)
    if spec.get("loader") == "parquet_server":
        return _download_via_parquet_server(name, out_dir)

    logger.info("Chargement de %s (%s)", spec["hf_id"], name)
    if spec.get("config"):
        ds = load_dataset(spec["hf_id"], spec["config"], token=load_hf_token())
    else:
        ds = load_dataset(spec["hf_id"], token=load_hf_token())

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    paths: dict[str, Path] = {}
    for split, subset in tqdm(ds.items(), desc=name):
        dest = out_dir / f"{split}.parquet"
        subset.to_parquet(dest)
        paths[split] = dest
        logger.info("  %s -> %s (%d lignes)", split, dest, len(subset))

    return paths


def download_all() -> None:
    """Télécharge l'ensemble des datasets déclarés dans la configuration."""
    for name in DATASETS:
        download_dataset(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_all()
