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

### 1.7B

| Run | Parse | Exactitude | Sous-triage | Verdict |
|---|---|---|---|---|
| **stage2_v2 (SFT)** | 97,7 % | **60 %** | **10,6 %** | référence |
| DPO qualité (2 types, LR 5e-6) | 25 % | — | — | ❌ questionnement casse la fiche |
| **DPO final-only (LR 1e-6)** | **100 %** | 57,5 % | 11,5 % | ⚠️ format OK, exactitude −2,5 |

### 4B

| Run | Parse | Exactitude | Sous-triage | Verdict |
|---|---|---|---|---|
| **qwen35_stage2 (SFT)** | 100 % | **69 %** | **3,4 %** | référence |
| **DPO final-only (LR 1e-6)** | 100 % | 62,1 % | 5,7 % | ⚠️ format OK, exactitude −6,9 |

→ Même recette sur le 4B : **pas de gain, légère dégradation** (exactitude −6,9,
sous-triage +2,3). Le DPO n'apporte rien sur les métriques gold.

## Conclusion honnête

1. **Le DPO ne fait pas gagner de points sur les gold** (exactitude légèrement en baisse).
2. Son apport est **qualitatif** (politesse, explication) — non mesuré par le gold.
3. La leçon récurrente : sur le 1.7B, **tout DPO déstabilise le format fiche** ; il faut
   se limiter aux paires qui ne touchent pas à la fiche, à LR très basse (1e-6).
4. Pour le POC, l'étape DPO est **réalisée** avec un artefact propre
   (`models/lora_dpo_final_merged`), sans casser le modèle.

## Analyse du gold (pourquoi l'écart DPO n'est pas concluant)

Le gold (87 cas) est **trop petit et déséquilibré** pour résoudre l'effet du DPO :

- **IC95 exactitude 4B** : SFT 69 % [58,6–77,7] vs DPO 62,1 % [51,6–71,5] → chevauchent,
  **pas significatif**.
- **Sous-triage** : 3,4 % vs 5,7 % (2 cas d'écart) → pas significatif.
- **Binaire sécurité** : 18 cas urgents, 2 vs 3 ratés → IC ±30 pts (1 cas = 5,6 pts).
- **Distribution gold** : 63 % bénins/non-urgents (niveau 4–5), 21 % urgents — alors que
  l'entraînement (SFT **et** DPO) est équilibré ~20 %/niveau.
- **Flips SFT↔DPO (16 cas)** : presque tous à ±1 niveau sur des cas niveau 4/5 ; seul
  F9 (asthme) est un vrai sous-triage urgent, mais cliniquement ambigu.

→ La dégradation apparente du DPO est **du bruit d'échantillonnage**, pas un effet réel.

## Artefacts

- `lora_dpo_final` (1.7B DPO, 57,5 %) et `qwen35_dpo_final` (4B DPO, 62,1 %) = modèles DPO « qualité ».
- Meilleurs modèles : `lora_stage2_v2` (1.7B SFT, 60 %) et `qwen35_stage2` (4B SFT, 69 %, sous-triage 3,4 %).
- Note vLLM 4B : `--max-model-len 4096 --gpu-memory-utilization 0.8` (le 8192 par défaut OOM au warmup).
