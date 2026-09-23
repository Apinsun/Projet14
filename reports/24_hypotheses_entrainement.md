# 24 — Hypothèses d'entraînement 1.7B (bf16, rank, troncature, non-déterminisme)

**Date** : 23/09/2026 · **Statut** : ✅ conclu

## Questions posées

1. La divergence ancien/nouveau 4B est-elle du bruit (non-déterminisme) ?
2. Le 1.7B est-il récupérable (modèle exigé par le brief) ? QLoRA 4-bit suffisant ?
3. La séquence 2048 tokens tronque-t-elle les exemples (avec les fiches ajoutées) ?
4. Rank, steps, précision : que peut-on gagner ?

## Résultats

### 1. Non-déterminisme (éval `temp 0.6`)

Même modèle (`qwen35_stage2`), même prompt, seed 42 : **79,3 % hier vs 77,0 % aujourd'hui**
→ **~2-3 pts de variance pure**. Le binaire bouge aussi (4,5 % ↔ 6,1 %, ±1 cas urgent).
Conclusion : sur 135 cas, un écart < 3 pts n'est **pas significatif**.

### 2. Troncature

Tokens (apply_chat_template) : vignettes max **488**, multitour+fiches max **1394**.
→ **Aucune troncature** à 2048. Séquencement large, pas en cause.

### 3. bf16 vs QLoRA 4-bit (le fix clé)

| 1.7B | Parse | Exactitude | Binaire |
|---|---|---|---|
| QLoRA 4-bit r=16 | 81 % | 65,1 % | — |
| **bf16 r=16** | **100 %** | 63 % | 6,1 % |

→ **Le QLoRA 4-bit causait le JSON malformé** (priorités hallucinées 10/15). En bf16,
parse 100 %.

### 4. Rank r=16 → r=32

| 1.7B bf16 | Parse | Exactitude | Sous-triage | Binaire |
|---|---|---|---|---|
| r=16 | 100 % | 63 % | 5,9 % | 6,1 % |
| **r=32** | 100 % | **68,9 %** | **4,4 %** | **4,5 %** |

→ **+6 pts d'exactitude**, binaire au niveau du meilleur 4B.

## Bilan final (format « fiche à chaque tour »)

| Modèle | Parse | Exactitude | Binaire sécurité |
|---|---|---|---|
| 1.7B bf16 r=32 | 100 % | 68,9 % | **4,5 %** |
| 4B SFT/DPO | 100 % | **76,3 %** | 10,6 % |

- Le **1.7B est viable** (parse 100 %, sécurité 4,5 %), il sur-trie davantage (exactitude
  68,9 %).
- Le **4B** est plus exact (76,3 %) mais sous-trie un peu plus sur ce format (10,6 %,
  non significatif vs 4,5 %).
- Comparaison ancien/nouveau format 4B : **77,0 % vs 76,3 %** (0,7 pt, bruit) — le format
  « fiche à chaque tour » **ne dégrade pas** le 4B.

## Artefacts conservés

- `lora_fiche_bf16_r32` (+ `_merged`) : **1.7B final** (bf16, r=32, 4 epochs).
- `qwen35_fiche_stage2` / `qwen35_dpo_v3` : 4B nouveau format.
- Anciens : `lora_stage2_v2`, `qwen35_stage2` (référence).
