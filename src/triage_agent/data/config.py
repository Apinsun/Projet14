"""Configuration centrale des sources de données du projet.

Chaque entrée du registre :class:`DATASETS` décrit un corpus utilisé par le projet :

- ``hf_id`` : identifiant Hugging Face Hub ;
- ``language`` : langue dominante du corpus (``fr`` / ``en``) ;
- ``role`` : usage prévu dans le pipeline (``SFT`` pour les paires
  instruction-réponse, ``DPO`` pour les paires de préférences).

Les chemins de stockage sont centralisés ici : les données brutes vont dans
``data/raw`` (dossier ignoré par Git) et les données transformées dans
``data/processed``.
"""

from __future__ import annotations

from pathlib import Path

# Racine du dépôt (deux niveaux au-dessus de ce fichier : src/triage_agent/data).
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ordre déterministe d'affichage / traitement.
DATASETS: dict[str, dict] = {
    "frenchmedmcqa": {
        "hf_id": "qanastek/frenchmedmcqa",
        "language": "fr",
        "role": "SFT",
        "description": (
            "3 105 QCM issus des examens officiels de pharmacie (France), "
            "5 options A-E, réponse unique ou multiple."
        ),
    },
    "medquad": {
        "hf_id": "lavita/MedQuAD",
        "language": "en",
        "role": "SFT",
        "description": (
            "≈47k paires question/réponse anglaises extraites des instituts NIH "
            "(GHR, Genetics Home Reference), couvrant 37 types de questions."
        ),
    },
    "mediqa": {
        "hf_id": "bigbio/mediqa_qa",
        "config": "mediqa_qa_bigbio_qa",
        "loader": "parquet_server",
        "language": "en",
        "role": "SFT",
        "description": (
            "Tâche officielle MediQA 2019 (question answering) : paires "
            "question patient -> réponse clinique (config bigbio_qa)."
        ),
    },
    "ultramedical_preference": {
        "hf_id": "TsinghuaC3I/UltraMedical-Preference",
        "language": "en",
        "role": "DPO",
        "description": (
            "Paires de préférences médicales (chosen vs rejected) générées par "
            "plusieurs modèles, avec feedback justifiant le classement."
        ),
    },
}
