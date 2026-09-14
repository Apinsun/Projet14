# AGENTS.md — conventions de travail pour les agents IA

Ce fichier s'adresse aux agents IA (et aux humains) travaillant sur ce dépôt.
À lire en début de session.

## Projet

POC d'un agent IA de **triage médical** (CHSA) : fine-tuning de **Qwen3-1.7B** par
**SFT (LoRA)** puis **DPO**, déploiement **vLLM/FastAPI/Docker** + CI/CD.
Protocole de triage : **FRENCH (SFMU, 5 niveaux)**. Livrable attendu sur ~4 semaines.

## Règle d'or : l'index des rapports

- Toute la documentation de travail vit dans `reports/`, indexée dans **`reports/README.md`**.
- **Chaque création ou modification de rapport DOIT mettre à jour `reports/README.md`**
  (statut, description, date). Ne jamais l'oublier.
- En début de session, **lire `reports/README.md`** pour savoir ce qui existe déjà,
  puis `reports/05_plan_attaque.md` pour la stratégie.

## Environnement

- **Python 3.12 obligatoire** (le 3.14 système est incompatible avec l'écosystème ML).
  Python fourni via `uv` ; dépendances gérées par **poetry** (`poetry run ...`).
- **vLLM** tourne dans un venv dédié `.venv-vllm` (Python 3.12), séparé du venv poetry ;
  il dialogue avec le code par HTTP (`/v1/chat/completions`). Voir
  `reports/11_vllm_troubleshooting.md` pour les pièges (ninja, nvcc, VRAM).
- **Données** : tout est sous `data/` (**gitignoré**, sauf `src/triage_agent/data/`).
- **Pas de notebooks** : des scripts Python dans `scripts/` + des rapports Markdown.
- Linter : `poetry run ruff check src scripts` (doit passer avant tout commit).

## Décisions clés (ne pas re-débattre sans raison)

1. **Niveau de triage toujours calculé** (règle FRENCH), jamais deviné par un LLM.
   Le LLM ne sert qu'à *habiller* en langage naturel.
2. **Format de sortie agent** : raisonnement interne dans les balises natives Qwen3
   `<think>...</think>` + fiche structurée `[FICHE] {"name":"finalize_triage","arguments":{...}} [/FICHE]`
   + explication patient. Le `<think>` est masqué au patient (via `reasoning_parser` vLLM)
   mais conservé pour audit.
3. **Incertitude** : plage d'urgence `[borne_urgente, borne_bénigne]` → retenir la
   borne prudente ; un red flag non écarté = présent.
4. **Entraînement** : **Unsloth** (SFT + DPO). Le LoRA final est **fusionné** dans les
   poids avant serving vLLM (pas de `--enable-lora`).
5. **Inférence** : **vLLM** (exigé par le brief), pas Ollama. Ollama/Leila ne sert qu'à
   **générer** les données synthétiques.
6. **Évaluation** : 3 étages (5 niveaux synthétique / 4-3 niveaux gold / binaire de
   sécurité), métrique principale = **sous-triage** (pondéré par la distance).
   Les outputs d'éval sont persistés dans `data/processed/gold/results/`.
7. **Honnêteté** : expliquer franchement les limites ; toute conclusion doit pouvoir
   être **validée par un humain** (revue d'échantillons).

## Conventions de travail

- Committer souvent, avec des messages clairs et un périmètre cohérent.
- Les artefacts lourds (données, modèles, logs) vont dans `/tmp` ou `data/` (gitignoré),
  jamais dans Git.
- Toute nouvelle étape produit un **rapport** + met à jour l'index + (si pertinent)
  `reports/05_plan_attaque.md`.

## État actuel (à mettre à jour)

- ✅ Étape 1 (données de base) + dataset de triage synthétique (600 exemples).
- ✅ Jeux gold téléchargés + harness d'évaluation + baseline vLLM (0/87).
- ⏭️ Reste : DPO triage (optionnel) → SFT Unsloth (semaine 2) → DPO → déploiement.
