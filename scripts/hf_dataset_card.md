---
license: cc-by-4.0
language:
  - fr
task_categories:
  - text-generation
pretty_name: Triage médical français (FRENCH — SFMU)
size_categories:
  - 1K<n<10K
tags:
  - medical
  - triage
  - french
  - synthetic
  - sft
  - dpo
---

# Triage médical français (FRENCH — SFMU)

Dataset **synthétique** pour entraîner un agent conversationnel de **triage médical** aux
urgences, fondé sur la grille **FRENCH** (SFMU, 5 niveaux d'urgence).

⚠️ **Ce dataset est synthétique** : il a été généré par un LLM à partir de règles
déterministes, **sans validation clinique**. Il est destiné à la **recherche uniquement**,
et non à un usage médical en production.

## Construction

1. La grille **FRENCH** (196 règles, 16 catégories) sert de source de vérité : chaque cas a
   un niveau d'urgence **correct par construction**.
2. Un **squelette de dialogue** (rôles + questions attendues) est dérivé de la règle.
3. Un LLM local (Qwen3.8-27B, GGUF Q4_K_M via Ollama) transforme ce squelette en **dialogue
   naturalisé**, en produisant aussi le raisonnement interne (`<think>`) et la fiche
   structurée (`<FICHE>`).

## Contenu

| Fichier | Type | Exemples | Rôle |
|---|---|---|---|
| `data/sft_vignettes.jsonl` | mono-tour | 300 | SFT : évaluer et conclure immédiatement |
| `data/sft_multiturn_fiche_nat.jsonl` | multi-tours | 1 074 | SFT : questionner avant de conclure |
| `data/dpo_quality_v3.jsonl` | paires chosen/rejected | 300 | DPO : préférer la forme polie à la forme brusque |

## Format

Chaque ligne est un objet JSON.

- Les fichiers **SFT** ont un champ `messages` (rôles `system` / `user` / `assistant`).
- Le fichier **DPO** a trois champs : `prompt`, `chosen`, `rejected`.

L'agent répond dans ce format :

- `<think>…</think>` : raisonnement interne (masqué au patient, conservé pour audit) ;
- `<FICHE>{…}</FICHE>` : fiche structurée (JSON) destinée au système d'information ;
- texte en langage naturel : l'explication montrée au patient.

## Statistiques

- Langue : **français** ;
- **1 674 exemples** au total (300 + 1 074 + 300) ;
- niveaux FRENCH 1–5 répartis sur 16 catégories cliniques.

## Limitations

- **Synthétique**, généré par un LLM, **non validé cliniquement** ;
- le patient ne fournit que du « patient-reportable » (symptômes, intensité, durée,
  antécédents) — aucune constante mesurée (PAS, SpO₂, ECG).

## Licence

CC-BY-4.0. Usage de recherche uniquement ; pas d'usage clinique.
