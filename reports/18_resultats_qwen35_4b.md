# 18 — Résultats SFT Qwen3.5-4B

**Date** : 22/09/2026 · **Statut** : ✅ (stage1+stage2 entraînés, évalués)

## Objectif

Tester notre recette SFT (base → triage multi-tours) sur un modèle plus gros
(**Qwen3.5-4B**) et mesurer le gain vs Qwen3-1.7B.

## Run

| Paramètre | Valeur |
|---|---|
| Modèle | `Qwen/Qwen3.5-4B` (Instruct) |
| Venv | `.venv-train-4b` (transformers 5.5.0, torch 2.12.1 cu130) |
| LoRA | **bf16** (pas QLoRA), r=16, `target_modules="all-linear"` |
| Stage1 | base 4 500, 1 epoch, 563 steps, loss finale 0,76 |
| Stage2 | triage 1 374, 2 epochs, 344 steps (reprise stage1) |
| Sortie | `models/qwen35_stage2` + `_merged` (8,7 Go) |
| Serving | vLLM + `--language-model-only --reasoning-parser qwen3` |

## Résultats mono-tour (gold FR, zero-shot)

| Métrique | Qwen3-1.7B (stage2_v2) | **Qwen3.5-4B** | Delta |
|---|---|---|---|
| Parse `<FICHE>` | 97,7 % (85/87) | **100 %** (87/87) | +2,3 |
| Exactitude | 60 % | **69 %** | **+9** |
| **Sous-triage** | 10,6 % | **3,4 %** | **−7,2** 🎯 |
| Binaire (urgences) | 16,7 % | **11,1 %** | −5,6 |
| Sur-triage | 29,4 % | 27,6 % | −1,8 |

→ **Hypothèse confirmée** : un modèle de base plus intelligent améliore nettement le
triage (exactitude +9 pts, sous-triage divisé par 3).

## Mode interactif : comportement nuancé

| Cas | Qwen3-1.7B | Qwen3.5-4B |
|---|---|---|
| « mal au poignet » (léger) | questionne | **questionne** ✓ |
| « douleurs **intenses** au ventre » | questionne systématiquement | **finalise directement** (priority 2 + red flag) |

Le 4B est **plus « clinique »** : sur un symptôme alarmant (douleur intense), il
reconnaît le red flag et conclut immédiatement — ce qui est **défendable médicalement**,
mais diffère du questionnement systématique du 1.7B. Il questionne bien sur les
symptômes bénins.

**Conséquence pratique** : en prod/démo, le 4B questionne pour les cas non urgents et
trie immédiatement les urgences — c'est un comportement d'agent plus mature.

## Points de vigilance / apprentissages

1. **Sampling** : Qwen3.5 recommande `temp=1.0 + presence_penalty=1.5` (le `temp=0.6`
   de Qwen3 rendait le 4B plus « rigide »). À adopter pour le 4B.
2. **`--reasoning-parser qwen3`** : sépare proprement `<think>` (champ `reasoning`) du
   contenu — gratuit, c'est exactement notre besoin « masquer au patient, garder pour audit ».
3. **DeltaNet sans kernel rapide** : `flash-linear-attention` non installé → repli torch
   (training ~1,3-3 s/step, acceptable). Installable si on veut accélérer.
4. **Stage3 inutile** : sur le 1.7B, la continuation +1 epoch a sur-appris (exactitude
   60→51 %). On s'arrête à 2 epochs.

## Prochaine étape : DPO

Le sous-triage résiduel (3,4 % sur le 4B) + le questionnement à affiner → **DPO**
(chosen = prudent/questionnant, rejected = sous-triage/fiche prématurée).
