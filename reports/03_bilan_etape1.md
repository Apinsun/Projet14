# Étape 1 — Bilan complet : préparation et structuration des données

**Projet** : POC d'un agent IA de triage médical (CHSA) — fine-tuning Qwen3-1.7B (SFT + DPO)
**Périmètre** : Semaine 1 — collecte, nettoyage, standardisation, anonymisation, constitution des jeux SFT/DPO/évaluation
**Date** : 2026-09-08

---

## 1. Résumé exécutif

L'étape 1 est **terminée et reproductible**. À partir des 4 corpus recommandés
(FrenchMedMCQA, MedQuAD, MediQA, UltraMedical-Preference), nous avons produit un
**corpus médical bilingue nettoyé, standardisé, anonymisé et découpé**, prêt pour
l'entraînement :

| Jeu | Volume | Rôle |
|---|---|---|
| `sft_train.jsonl` | 4 500 | SFT (2 160 FR + 2 340 EN) |
| `sft_val.jsonl` | 500 | Validation SFT (240 FR + 260 EN) |
| `dpo_train.jsonl` | 5 000 | Alignement DPO (EN) |
| `clinical_eval.jsonl` | 1 038 | Évaluation clinique **isolée** |

Le **50/50 FR-EN est atteint** (2 400 / 2 600 sur le SFT), le jeu d'évaluation
clinique est **gelé avant tout entraînement** (anti-fuite), et l'analyse
d'anonymisation a conclu à **l'absence de données personnelles** dans ces corpus
publics (voir §5).

---

## 2. Démarche générale

Le pipeline a été construit en **5 phases**, chacune scriptée, traçable et
rejouable :

```
téléchargement → EDA → standardisation/nettoyage → sélection + anonymisation → découpage
```

- **Téléchargement** (`scripts/download_datasets.py`) : les 4 sources vers `data/raw/` (Parquet).
- **EDA** (`scripts/run_eda.py`) : comprendre schéma, volumes, qualité, langues → rapport 01.
- **Standardisation** (`scripts/transform_datasets.py`) : chaque source → schéma JSONL commun.
- **Sélection + anonymisation** (`scripts/run_pipeline.py`) : constitution SFT/DPO/éval + Presidio.
- **Revue** (`scripts/run_review.py`) : contrôle post-traitement par source → rapport 02.

**Principe directeur** : *prioriser la qualité et l'auditabilité sur la quantité*,
conformément aux recommandations du sujet.

---

## 3. Ce qui a été fait (et pourquoi)

### 3.1 Mise en place technique

- **Python 3.12** provisionné via `uv` (le 3.14 système n'est pas compatible avec
  l'écosystème PyTorch/transformers à venir). `uv` télécharge un Python autonome
  **sans sudo**.
- **Poetry** pour la gestion des dépendances (package `triage_agent`, layout `src/`).
- **`data/` gitignoré** (données locales), `.env` avec `HF_TOKEN` (jamais commité).
- Deux écueils contournés : `datasets` 5.x ne supporte plus les scripts de
  chargement → FrenchMedMCQA parsé depuis son ZIP source, MediQA récupéré via les
  parquets convertis du *datasets-server*.

### 3.2 Analyse exploratoire (EDA)

Objectif : connaître chaque source avant de la transformer. Constats clés :

- **FrenchMedMCQA** : 3 105 QCM pharmacie (FR), propre, **seule source FR**.
- **MedQuAD** : 47 441 lignes mais **31 034 réponses nulles** dans le miroir
  `lavita` (sources ADAM/MPlusDrugs vides) → **16 407 paires réellement exploitables**.
- **MediQA** : deux variantes ; la plus répandue (`medalpaca/medical_meadow_mediqa`)
  n'a que **154 questions uniques** (données de *ranking* de réponses).
- **UltraMedical-Preference** : 112 362 paires `chosen/rejected` (EN), 32 829 prompts dupliqués.

### 3.3 Standardisation et nettoyage

Chaque source → schéma JSONL commun (`messages` pour SFT, `prompt/chosen/rejected`
pour DPO), avec nettoyage : suppression HTML, normalisation des espaces, suppression
des citations `[1]`, filtres de longueur (question ≤ 2 000, réponse ≤ 4 000).

| Source | Brut | Standardisé | Transformation |
|---|---|---|---|
| FrenchMedMCQA | 3 105 | 3 105 | QCM → instruction (question + options) / réponse rédigée |
| MedQuAD | 47 441 | 14 559 | filtre réponses non nulles + dédup + question→réponse |
| MediQA | 383 | 319 | extraction réponse « gold » (ReferenceRank=1) |
| UltraMedical-Preference | 112 362 | 67 429 | dédup prompts + extraction contenu assistant |

### 3.4 Anonymisation RGPD (Presidio)

- Moteur Presidio avec modèles spaCy **FR** (`fr_core_news_md`) et **EN**
  (`en_core_web_md`).
- **Constat** : l'analyse (échantillons + types d'entités détectées) montre que les
  entités signalées par Presidio sont des **faux positifs médicaux**
  (`PERSON` = éponymes de maladies, `URL` = « E.coli », `MEDICAL_LICENSE` =
  identifiants SNP `rs1063192`, `UK_NHS` = n° de centre antipoison…).
- **Décision** : ne masquer que les identifiants **sans ambiguïté** (email, IBAN,
  carte bancaire, IP). Résultat : **0 donnée personnelle réelle** dans le corpus final.
- Tout est **tracé** dans `data/processed/anonymized/audit.json` (entités détectées
  vs masquées), ce qui constitue la preuve documentaire du contrôle RGPD.

### 3.5 Constitution des jeux et découpage

- **SFT** ≈ 5 000 : **2 400 FR** (FrenchMedMCQA train+validation, test exclu) +
  **2 600 EN** (MedQuAD + MediQA train/validation).
- **DPO** : 5 000 paires UltraMedical, échantillonnées de façon **stratifiée** sur
  `label_type` (hard/length/easy).
- **Évaluation clinique isolée** : FrenchMedMCQA `test` (622) + MediQA `test` (116)
  + réserve MedQuAD (300) — **jamais croisée avec l'entraînement**.
- **Split SFT** train/val (90/10) stratifié langue × source.

---

## 4. Résultats chiffrés

### 4.1 Volumes par source (cheminement complet)

| Source | Brut | Exploitable | Standardisé | SFT | DPO | Éval |
|---|---|---|---|---|---|---|
| FrenchMedMCQA | 3 105 | 3 105 | 3 105 | 2 400 | — | 622 |
| MedQuAD | 47 441 | 16 407 | 14 559 | 2 570 | — | 300 |
| MediQA | 383 | 383 | 319 | 30 | — | 116 |
| UltraMedical-Preference | 112 362 | — | 67 429 | — | 5 000 | — |

### 4.2 Jeux finaux

| Fichier | Lignes | FR | EN |
|---|---|---|---|
| `sft_train.jsonl` | 4 500 | 2 160 | 2 340 |
| `sft_val.jsonl` | 500 | 240 | 260 |
| `dpo_train.jsonl` | 5 000 | 0 | 5 000 |
| `clinical_eval.jsonl` | 1 038 | 622 | 416 |

---

## 5. Points forts

1. **Reproductibilité totale** : seed fixe (42), manifestes à chaque étape
   (`manifest.json`, `audit.json`), pipeline rejouable en une commande.
2. **Discipline anti-fuite** : l'évaluation clinique est gelée avant toute
   manipulation ultérieure — le benchmark final sera non biaisé.
3. **Honnêteté sur l'anonymisation** : pas de sur-masquage qui aurait dégradé le
   contenu médical ; la conclusion « 0 PII » est documentée et défendable.
4. **Traçabilité** : chaque enregistrement porte `source`, `source_id`, `split`,
   `license`, `lang` → audit complet possible.
5. **Formats directement exploitables** : schémas `messages` et
   `prompt/chosen/rejected` compatibles avec `SFTTrainer` et `DPOTrainer` (TRL).
6. **Robustesse technique** : contournement propre des scripts de chargement retirés
   dans `datasets` 5.x, sans dépendre de fonctionnalités dépréciées.

---

## 6. Points faibles et limites

1. **MedQuAD incomplet** (miroir `lavita`) : 31 k réponses vides, et **artefacts
   source** (mots collés autour d'italiques : « invaderssuch », apostrophes
   manquantes « Hashimotos »). Accepté tel quel, mais à savoir.
2. **MediQA minuscule et bruité** : 319 paires, certaines réponses « gold » ont un
   `ReferenceScore` faible (2/4). Contribution négligeable au volume EN.
3. **UltraMedical = science biomédicale large**, pas du triage clinique. Le DPO
   alignera donc surtout la *qualité/style* des réponses, pas le *domaine* triage.
4. **Métadonnées cliniques vides** : `symptoms`, `medical_history`, `vital_signs`
   sont des placeholders — les corpus généralistes ne sont **pas annotés triage**.
   C'est une limite structurelle vis-à-vis du cas d'usage CHSA.
5. **FR au plafond** : le 50/50 consomme ~97 % de FrenchMedMCQA (hors test). Aucune
   marge pour un jeu de validation FR dédié supplémentaire.
6. **Pas de versionnement distribué** : les données sont locales (`data/` gitignoré).
   Pour le livrable « dataset versionné », il faudra pousser sur Hugging Face Hub
   (dépôt gated) ou DVC.

---

## 7. Choix structurants et impact sur la suite

| Choix | Décision retenue | Impact sur l'étape 2+ |
|---|---|---|
| **Source MediQA** | `bigbio/mediqa_qa` (gold ReferenceRank=1) | Source propre mais marginale ; pas de risque de contamination ni de sur-coût |
| **Miroir MedQuAD** | `lavita/MedQuAD` (16,4 k exploitables) | Suffisant pour 2 600 EN ; si besoin de plus d'EN, changer de miroir |
| **Ratio FR/EN** | 2 400 / 2 600 | Modèle exposé au FR dans une proportion proche de l'usage cible ; pas de marge FR |
| **Anonymisation** | masquer uniquement identifiants sans ambiguïté | Corpus final intact ; posture RGPD documentée (audit) |
| **Format des données** | `messages` (SFT) / `prompt-chosen-rejected` (DPO) | Compatible nativement TRL → gain de temps semaine 2 |
| **Évaluation gelée** | test natifs + réserve MedQuAD | Benchmark non biaisé pour mesurer les vrais progrès |
| **Taille DPO** | 5 000 (configurable) | Suffisant pour un POC ; rejouable pour 10 000 si besoin |
| **Filtres de longueur** | question ≤ 2 000, réponse ≤ 4 000 chars | Tient dans la fenêtre de contexte de Qwen3-1.7B (32 k tokens) |

---

## 8. Décisions restant à prendre avant l'étape 2

1. **Taille du DPO** : maintenir 5 000 ou passer à 10 000 ?
2. **Métriques d'évaluation clinique** et **seuils d'acceptation** : exact-match sur
   les QCM ? score LLM-as-judge ? précision sur le jeu MedQuAD/MediQA ? (exigé par le
   sujet, à définir avant l'entraînement).
3. **Outil de tracking** : MLflow ou Weights & Biases ?
4. **Versionnement du dataset** : pousser le corpus final sur le Hub HF (gated) pour
   le livrable, ou rester en local pour l'instant ?
5. **Éventuelle augmentation FR** (traduction d'une partie de MedQuAD) : repoussée,
   mais c'est le seul levier si le modèle s'avère faible en français.

---

## 9. Ce qu'il faudra faire pour l'étape 2 (SFT LoRA)

1. **Environnement d'entraînement** : ajouter `torch`, `transformers`, `peft`,
   `trl`, `accelerate`, `bitsandbytes` (et éventuellement `unsloth`), + `mlflow`/`wandb`.
2. **Préparation du modèle** : Qwen3-1.7B-Base + *chat template* adapté au format
   `messages` de nos données.
3. **Petits runs LoRA d'abord** : valider la pipeline (1 epoch, petit batch) avant
   de monter en charge — recommandé par le sujet.
4. **Hyperparamètres + seeds documentés**, checkpoints intermédiaires pour reprise.
5. **Évaluation intermédiaire** sur `sft_val` puis sur `clinical_eval` (jamais en
   entraînement), avec les métriques définies au §8.
6. **Contrôles de sécurité** : hallucinations, recommandations dangereuses (le sujet
   l'exige avant validation).
7. **Itération** : ajuster LoRA (r, alpha, lr), puis figer un checkpoint pour le DPO
   (semaine 3).

---

## 10. Reproductibilité

Tout le pipeline se rejoue depuis zéro :

```bash
poetry install
poetry run python scripts/download_datasets.py      # → data/raw/
poetry run python scripts/run_eda.py                 # → reports/eda_results.json
poetry run python scripts/plot_eda.py                # → reports/figures/
poetry run python scripts/transform_datasets.py      # → data/processed/*.jsonl
poetry run python scripts/run_pipeline.py            # → data/processed/{selected,anonymized,final}/
poetry run python scripts/run_review.py              # → reports/02_preparation_donnees.md
poetry run python scripts/inspect_data.py sft_train 3  # inspection manuelle
```

Les tailles (FR/EN/DPO) et le seed sont modifiables en tête de
`src/triage_agent/data/build.py`.

---

## 11. Annexe — Fichiers clés

| Chemin | Contenu |
|---|---|
| `reports/01_analyse_datasets.md` | EDA détaillée des 4 sources |
| `reports/02_preparation_donnees.md` | Revue post-traitement par source |
| `data/processed/manifest.json` | Stats de standardisation |
| `data/processed/selected/manifest.json` | Stats de sélection |
| `data/processed/anonymized/audit.json` | Journal d'audit RGPD (entités détectées/masquées) |
| `data/processed/final/manifest.json` | Volumes des jeux finaux |
| `src/triage_agent/data/` | Modules du pipeline (config, download, clean, transform, build, anonymize, pipeline, review) |
