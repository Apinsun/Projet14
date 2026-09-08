# Projet 14 — POC d'un agent IA de triage médical (CHSA)

Proof of Concept d'un agent conversationnel de triage médical pour le service des
urgences du Centre Hospitalier Saint-Aurélien (CHSA), basé sur le modèle
**Qwen3-1.7B-Base** affiné par **SFT (LoRA)** puis aligné par **DPO**, déployé
derrière une API **FastAPI** avec **vLLM** et un pipeline **CI/CD (GitHub Actions)**.

## Objectif

L'agent doit :

- collecter les symptômes du patient via un questionnaire adaptatif ;
- évaluer le niveau de priorité (urgence maximale / modérée / différée) ;
- fournir des explications claires sur l'évaluation et les recommandations ;
- garantir la traçabilité de chaque interaction.

## Plan de la mission (4 semaines)

1. **Semaine 1** — Préparation et structuration des données (corpus bilingue
   FR/EN, ~5 000 paires SFT, jeu DPO, anonymisation RGPD via Presidio).
2. **Semaine 2** — Fine-tuning supervisé (SFT) de Qwen3-1.7B avec LoRA.
3. **Semaine 3** — Alignement par préférences (DPO).
4. **Semaine 4** — Déploiement (vLLM + FastAPI + Docker) et validation.

## Sources de données

| Corpus | Hub HF | Langue | Usage |
|---|---|---|---|
| FrenchMedMCQA | `qanastek/frenchmedmcqa` | FR | SFT |
| MedQuAD | `lavita/MedQuAD` | EN | SFT |
| MediQA | `medalpaca/medical_meadow_mediqa` | EN | SFT |
| UltraMedical-Preference | `TsinghuaC3I/UltraMedical-Preference` | EN | DPO |

## Structure du projet

```
.
├── Docs/                  # Énoncé du projet et notes
├── data/                  # Données locales (ignorées par Git)
│   ├── raw/               #   Datasets bruts (Parquet)
│   └── processed/         #   Données standardisées / anonymisées
├── scripts/               # Scripts exécutables (téléchargement, EDA, pipeline)
├── src/triage_agent/      # Package Python (logique métier)
│   └── data/              #   Modules de préparation des données
├── reports/               # Rapports d'analyse (Markdown)
└── tests/                 # Tests unitaires
```

## Installation

```bash
# Python 3.12 requis (voir pyproject.toml)
poetry install
```

Le token Hugging Face est lu depuis le fichier `.env` (`HF_TOKEN=...`), ignoré
par Git.

## Usage

```bash
# Télécharger les datasets bruts
poetry run python scripts/download_datasets.py

# Analyse exploratoire
poetry run python scripts/run_eda.py
```
