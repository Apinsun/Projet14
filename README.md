# Agent IA de triage médical — POC (CHSA)

Proof of concept d'un agent conversationnel de **triage médical** pour les urgences,
basé sur **Qwen3-1.7B** affiné par **SFT (LoRA)** puis aligné par **DPO**, déployé derrière
**FastAPI + vLLM + Docker** avec un pipeline **CI/CD (GitHub Actions)**.

Le protocole de triage est **FRENCH** (SFMU, 5 niveaux d'urgence) : le niveau est **calculé**
par la règle, jamais deviné — le LLM conduit la discussion et produit une **fiche
structurée** (`<FICHE>`) pensée pour être parsée et intégrée au SI, un raisonnement interne
(`<think>`) conservé pour l'audit, et une explication en langage naturel pour le patient.

## Résultats clés

| Modèle | Fiches bien formées | Exactitude | Sous-triage (cas urgents) |
|---|---|---|---|
| **Qwen3-1.7B** (SFT LoRA bf16 r=32 + DPO) | **100 %** | **68,9 %** | **4,5 %** |

*(Évaluation sur un jeu gold externe de 135 cas : Levine + Ramaswamy + IyàwóBench.)*

## Dataset

Le dataset de triage est **synthétique** (généré depuis la grille FRENCH par un LLM 27B,
niveau correct par construction) et publié sur HuggingFace :

👉 **https://huggingface.co/datasets/apinsun/triage-french**

## Architecture

```
Patient ──► FastAPI (conteneur, port 8080) ──► vLLM (port 8000) ──► Qwen3-1.7B
                │                                   │
                └── journal d'audit (logs/audit.jsonl)
```

- **FastAPI** : conteneurisée (image publiée sur GHCR), API **stateless**, parsing
  `<think>`/`<FICHE>`, journal d'audit JSONL (traçabilité) ;
- **vLLM** : moteur d'inférence optimisé, modèle LoRA fusionné.

## Démarrage rapide

```bash
./start.sh     # lance vLLM (hôte) + l'app conteneurisée → http://localhost:8080
./stop.sh      # arrête tout
```

Interface web : **http://localhost:8080** · Documentation API : **http://localhost:8080/docs**

## Structure du projet

```
├── src/triage_agent/
│   ├── data/          # grille FRENCH, fiche, génération du dataset
│   ├── eval/          # harness d'évaluation sur le gold
│   ├── serving/       # API FastAPI (main, agent, vllm_client, audit)
│   └── parsing.py     # extraction <think>/<FICHE> (partagée eval + serving)
├── scripts/           # entraînement, évaluation, benchmark, publication HF
├── tests/             # 28 tests (unitaires + intégration, vLLM mocké)
├── Dockerfile         # image de l'app (deps de serving uniquement)
├── docker-compose.yml # option B : vLLM + app en conteneurs
├── .github/workflows/ # CI (lint/tests/build) + CD (push GHCR)
└── start.sh / stop.sh
```

## Entraînement & évaluation

```bash
poetry run python scripts/validate_dataset.py     # conformité du dataset avant re-train
./scripts/train.sh                                 # SFT 1.7B
./scripts/train-dpo.sh                             # DPO
poetry run python scripts/run_gold_eval.py         # évaluation sur le gold
poetry run python scripts/benchmark_latency.py     # benchmark de latence
```

## CI/CD

- **CI** (à chaque push) : lint (ruff) → 28 tests → build Docker + smoke test ;
- **CD** (sur tag `v*`) : build + publication de l'image sur **GHCR**, tirée au déploiement.

## Installation (développement)

Python 3.12, dépendances via poetry :

```bash
poetry install
```

Le token Hugging Face est lu depuis `.env` (non versionné).
