# Index des rapports

> **Référence unique du projet.** Chaque rapport est listé ici avec ce qu'il documente.
> **Règle** : toute création ou modification de rapport doit mettre à jour cet index
> (statut, contenu, date). Voir aussi `AGENTS.md`.

| # | Rapport | Ce qu'il documente | Statut |
|---|---|---|---|
| 01 | `01_analyse_datasets.md` | EDA des 4 corpus de base (FrenchMedMCQA, MedQuAD, MediQA, UltraMedical) | ✅ |
| 02 | `02_preparation_donnees.md` | Nettoyage + standardisation → `data/processed/*.jsonl` | ✅ |
| 03 | `03_bilan_etape1.md` | Bilan semaine 1 : jeux SFT/DPO/éval, anonymisation, leçons | ✅ |
| 04 | `04_protocole_french.md` | Le protocole de triage FRENCH (SFMU, 5 niveaux, 16 catégories) | ✅ |
| 05 | `05_plan_attaque.md` | **Stratégie globale** : décisions clés, architecture agent, plan SFT/DPO/éval/déploiement | ✅ |
| 06 | `06_spec_dataset_triage.md` | Spec détaillée du dataset de triage (schémas, format `<think>`/`[FICHE]`, validation) | ✅ |
| 07 | `07_construction_dataset_triage.md` | Construction du dataset synthétique (grille→fact sheets→vignettes→dialogues habillés), biais « Euh » | ✅ |
| 08 | `08_analyse_jeux_gold.md` | Jeux gold d'évaluation (Levine/Ramaswamy/IyàwóBench/psy) + **plan d'évaluation 3 étages** + mapping plage FRENCH | ✅ |
| 09 | `09_presentation_vllm.md` | Présentation de vLLM (moteur d'inférence), vs Ollama, installation, architecture | ✅ |
| 10 | `10_baseline_eval.md` | Baseline Qwen3-1.7B de base sur vLLM (0/87 zero-shot ; 6/87 few-shot greedy ; 46/87 few-shot params Qwen3) + installation vLLM | ✅ |
| 11 | `11_vllm_troubleshooting.md` | **Dépannage vLLM** : ninja, mismatch nvcc, VRAM, concurrence → checklist semaine 4 | ✅ |
| 12 | `12_plan_sft.md` | **Plan SFT (LoRA) Unsloth** : déroulement, dépendances, paramètres cruciaux, best practices Qwen3 + résultats du pilot | ✅ |
| 13 | `13_resultats_sft.md` | **Résultats SFT** : parcours des runs (pilot→full→2 étapes→continuation), découvertes (zero-shot>few-shot, drift `<FICHE>`), métriques finales | ✅ |
| 14 | `14_dataset_multitour_llm.md` | **Dataset multi-tours généré par LLM 27B** : 1074 dialogues (100 % corrects, 85 % LLM, niveaux équilibrés), mode tiers, pièges (`<think>` stop token, outage Ollama) | ✅ |
| 15 | `15_plan_sft_v2_multitour.md` | **Plan SFT v2** : données (4 500 base + 1 374 triage), stratégie 2 étapes, hyperparamètres, nombre de steps, éval | ✅ |
| 16 | `16_resultats_sft_v2.md` | **Résultats SFT v2 (multi-tour)** : parse 97,7 %, exactitude 60 %, sous-triage 10,6 % ; **mode interactif corrigé** | ✅ |
| 17 | `17_plan_qwen35_4b.md` | **Plan Qwen3.5-4B** : Instruct (pas Base), bf16 LoRA (pas QLoRA), transformers v5, target_modules all-linear, sampling | ✅ |
| 18 | `18_resultats_qwen35_4b.md` | **Résultats Qwen3.5-4B** : parse 100 %, exactitude 69 %, sous-triage 3,4 % (vs 10,6 % sur 1.7B) | ✅ |
| 19 | `19_dpo_tentative.md` | **DPO triage (tentative)** : pipeline OK mais non concluant (paires trop faciles, questionnement casse la fiche) | ⚠️ |
| 20 | `20_dpo_qualite.md` | **DPO qualité** (politesse/fiche/prise en charge) : final-only LR 1e-6, parse 100 %, exactitude 57,5 % (étape DPO réalisée) | ✅ |
| 21 | `21_gold_urgent.md` | **Gold urgent enrichi** (48 cas IyàwóBench REFER_NOW) : IC binaire sécurité ±15 → ±5,5 pts, SFT ≈ DPO confirmé | ✅ |
| 22 | `22_format_fiche_tour.md` | **Format fiche à chaque tour + patient naturalisé** (SFT v3) : 4B parse 100 %, 1.7B hors capacité | ✅ |
| 23 | `23_dpo_v3.md` | **DPO v3 (fiche à chaque tour)** : parse 100 %, DPO questionnement débloqué | ✅ |
| 24 | `24_hypotheses_entrainement.md` | **Hypothèses 1.7B** : bf16 (fix QLoRA), rank r=32, pas de troncature, variance ±3 pts | ✅ |

## Documents de travail (racine)

| Fichier | Rôle |
|---|---|
| `AGENTS.md` | Conventions de travail pour les agents IA (mettre à jour l'index, environnement, décisions) |
| `README.md` | Présentation générale du projet |
| `pyproject.toml` | Dépendances (poetry, Python 3.12) |

## Où sont les données / artefacts

| Chemin | Contenu |
|---|---|
| `data/raw/` | corpus bruts + jeux gold téléchargés (gitignoré) |
| `data/processed/` | jeux SFT/DPO/éval, dataset de triage, `gold/` (records + `results/`) |
| `src/triage_agent/` | code (data, eval) |
| `scripts/` | scripts exécutables (génération, préparation, évaluation) |
