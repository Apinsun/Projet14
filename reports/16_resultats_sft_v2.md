# 16 — Résultats SFT v2 (multi-tour)

**Date** : 22/09/2026 · **Statut** : ✅ (Stage2 entraîné + évalué)

## Contexte

SFT Stage2 entraîné sur le **dataset multi-tours** (rapport 14) pour corriger le mode
interactif. Reprise de l'adaptateur `lora_stage1` (base médicale 4 500), selon le plan
du rapport 15.

## Run

| Paramètre | Valeur |
|---|---|
| Données | 1 374 exemples (300 vignettes + 1 074 multi-tours) |
| Reprise | `models/lora_stage1` |
| Epochs | 2 (344 steps) |
| LR | 2e-4 cosine, warmup 5 |
| LoRA | r=16, QLoRA 4-bit, batch effectif 8, max_seq 2048 |
| Loss finale | ~0,5 |
| Sortie | `models/lora_stage2_v2` + `models/lora_stage2_v2_merged` (3,3 Go) |

## Résultats mono-tour (gold FR, zero-shot)

| Métrique | Stage3 (précédent) | **Stage2_v2** | Delta |
|---|---|---|---|
| Parse `<FICHE>` | 95 % | **97,7 %** (85/87) | +2,7 |
| Exactitude | 58 % | **60 %** | +2 |
| Sous-triage | 12 % | **10,6 %** | −1,4 |
| Binaire (urgences) | 16,7 % | 16,7 % | = |

→ Le dataset multi-tours **n'a pas dégradé** le mono-tour : il l'a légèrement amélioré.

## Résultat clé : le mode interactif est corrigé

Avant (bug) : `« j'ai des douleurs au ventre »` → fiche immédiate, sans question.
Maintenant :

```
👤 j'ai des douleurs intenses au ventre
🤖 Depuis combien de temps ces douleurs existent-elles ?
👤 depuis hier soir, plutôt à droite
🤖 Avez-vous des antécédents médicaux ?
👤 non, pas de fièvre, mais j'ai vomi deux fois
🤖 Quels médicaments prenez-vous habituellement ?
```

Et quand l'info est **complète** → le modèle conclut avec une fiche correcte (priority 4).

## Limites restantes

1. **`<think>` parfois vide** : le modèle émet parfois `<think></think>` vide (le
   raisonnement est appris mais pas toujours produit). Mineur (masqué au patient).
2. **Questions parfois redondantes** : le modèle suit la séquence durée→antécédents→
   traitements même si l'info est déjà donnée. Acceptable, à affiner (DPO).
3. **Sous-triage 10,6 %** (dont ~3 urgences/18 ratées) : reste le chantier **DPO**.

## Prochaine étape

**DPO** : paires de préférence triage (chosen = question avant fiche + triage prudent /
rejected = fiche prématurée + sous-triage) pour faire chuter le sous-triage vers ~0 et
affiner le questionnement.
