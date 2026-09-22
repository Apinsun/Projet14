# 20 — DPO « qualité de service » (politesse, fiche, prise en charge)

**Date** : 22/09/2026 · **Statut** : ✅ étape DPO réalisée (dégradation minime)

## Cadrage

Suite à l'impasse du DPO « sous-triage » (rapport 19), on a recentré la DPO sur la
**qualité de service** : politesse, questionnement, fiche correcte, explication de la
prise en charge. Le rejected ne diffère du chosen que sur la **forme**, jamais sur le
niveau de triage ni la présence de la fiche.

## Paires générées (27B)

2 types, la fiche restant programmatique (règle FRENCH) :

| Type | Chosen | Rejected |
|---|---|---|
| **Final** (150) | poli + fiche + explication | brusque + même fiche, sans explication |
| **Questionnement** (148) | question polie | brusque, sans question, sans fiche |

Générées par `scripts/generate_dpo_quality.py` (le 27B écrit le rejected, j'injecte la
fiche).

## Résultats des itérations (gold FR, zero-shot)

| Run | Parse | Exactitude | Sous-triage | Verdict |
|---|---|---|---|---|
| **stage2_v2 (SFT)** | 97,7 % | **60 %** | **10,6 %** | référence |
| DPO qualité (2 types, LR 5e-6) | 25 % | — | — | ❌ questionnement casse la fiche |
| **DPO final-only (LR 1e-6)** | **100 %** | 57,5 % | 11,5 % | ⚠️ format OK, exactitude −2,5 |

→ Le run « final-only, LR 1e-6 » (uniquement les paires de fiche/explication) **préserve
le format** (parse 100 %) avec une **dégradation minime** (exactitude −2,5 pts,
sous-triage +0,9 pt).

## Conclusion honnête

1. **Le DPO ne fait pas gagner de points sur les gold** (exactitude légèrement en baisse).
2. Son apport est **qualitatif** (politesse, explication) — non mesuré par le gold.
3. La leçon récurrente : sur le 1.7B, **tout DPO déstabilise le format fiche** ; il faut
   se limiter aux paires qui ne touchent pas à la fiche, à LR très basse (1e-6).
4. Pour le POC, l'étape DPO est **réalisée** avec un artefact propre
   (`models/lora_dpo_final_merged`), sans casser le modèle.

## Artefacts

- Conservé : `lora_dpo_final` (+ `_merged`) = modèle DPO « qualité », parse 100 %,
  exactitude 57,5 %.
- Meilleurs modèles : `lora_stage2_v2` (1.7B SFT, 60 %) et `qwen35_stage2` (4B, 69 %).
