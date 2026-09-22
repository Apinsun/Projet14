# 19 — DPO triage (tentative) — non concluant

**Date** : 22/09/2026 · **Statut** : ⚠️ pipeline OK, mais pas d'amélioration

## Objectif

Réduire le sous-triage du 1.7B (10,6 %) via DPO (chosen = prudent, rejected = sous-triage).

## Ce qui a été fait

- **Génération de paires** (`scripts/generate_dpo_pairs.py`) : 2 types —
  *sous-triage* (fiche correcte vs fiche rétrogradée) et *questionnement* (question vs fiche).
- **`scripts/train_dpo.py`** : DPOTrainer Unsloth (LoRA r=16, LR 1e-5/5e-6, beta 0.1),
  avec patch TRL 0.24 + transformers v5 (`is_*_available` renvoie un tuple).
- Mix UltraMedical (EN) + paires triage.

## Résultats (gold FR, zero-shot)

| Run | Parse | Exactitude | Sous-triage | Verdict |
|---|---|---|---|---|
| **stage2_v2 (SFT)** | 97,7 % | **60 %** | **10,6 %** | ✅ référence |
| DPO v1 (2 types + UM 2000) | **0 %** | — | — | ❌ fiche cassée |
| DPO v2 (sous-triage + UM 500) | 92 % | 53,7 % | 11,3 % | ❌ dégradé |
| DPO v3 (sous-triage ciblé, LR 5e-6) | 98,9 % | 54,7 % | 10,5 % | ⚠️ parse OK, exactitude −5 |

**Aucun run DPO n'améliore le modèle.** Le meilleur reste **stage2_v2 (SFT)**.

## Diagnostic

1. **Questionnement vs fiche** : les paires « question vs fiche » cassent le format
   (le modèle apprend à ne plus produire de fiche). À éviter ou fortement pondérer.
2. **Paires trop « faciles »** : le sous-triage programmatique (urgent → non-urgent)
   cible des erreurs que le modèle ne fait déjà pas. Ses vraies erreurs sont **subtiles**
   (+1 niveau, cas ambigus) — non couvertes par des paires déterministes.
3. **UltraMedical dilue** : 2000/500 paires EN sans fiche perturbent le format FR.
4. **Exactitude en baisse** : le DPO ajoute du « bruit » sans signal utile sur ces paires.

## Pistes (à creuser plus tard)

1. **Paires depuis les erreurs réelles** : faire tourner le modèle sur les gold, extraire
   ses sous-triages, et construire chosen/rejected à partir de ces cas précis.
2. **GRPO avec fonction de récompense** : récompenser directement `priority == true_level`
   et pénaliser `priority > true_level` (plus adapté qu'un DPO sur paires figées).
3. **Tuner beta/LR** + plus de paires triage (sans UltraMedical).

## État des artefacts

Les modèles DPO (non concluants) ont été **supprimés**. On conserve :
`lora_stage2_v2` (1.7B) et `qwen35_stage2` (4B, meilleur modèle global : exactitude 69 %,
sous-triage 3,4 %).
