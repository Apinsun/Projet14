# 25 — Comparatif des techniques (bf16, rank, DPO) — 1.7B et 4B

**Date** : 23/09/2026 · **Statut** : ✅ conclu

## Objectif

Tester systématiquement : QLoRA vs bf16, rank r=16 vs r=32, effet du DPO, sur le format
« fiche à chaque tour + patient naturalisé » (gold enrichi 135 cas, seed 42).

## Résultats complets

### 1.7B

| Config | Parse | Exactitude | Sous-triage | Binaire sécurité |
|---|---|---|---|---|
| QLoRA 4-bit r=16 (4 ep) | 81 % ❌ | 65,1 % | 6,4 % | — |
| bf16 r=16 (4 ep) | 100 % | 63 % | 5,9 % | 6,1 % |
| **bf16 r=32 (4 ep)** | 100 % | **68,9 %** | **4,4 %** | **4,5 %** |
| bf16 r=32 + DPO | 100 % | 68,9 % | 4,4 % | 4,5 % |

### 4B

| Config | Parse | Exactitude | Sous-triage | Binaire sécurité |
|---|---|---|---|---|
| bf16 r=16 SFT | 100 % | **76,3 %** | 5,9 % | 10,6 % |
| bf16 r=16 DPO | 100 % | 76,3 % | 5,9 % | 10,6 % |
| bf16 r=32 SFT | 100 % | 74,1 % | 6,7 % | 9,1 % |
| bf16 r=32 DPO | 100 % | 75,6 % | 5,9 % | 9,1 % |

### Ancien format (référence)

| Config | Parse | Exactitude | Binaire |
|---|---|---|---|
| 1.7B `lora_stage2_v2` | 97,7 % | 71,5 % | 9,5 % |
| 4B `qwen35_stage2` | 100 % | 77,0 % | 6,1 % |

## Conclusions claires

1. **bf16 répare le 1.7B** : QLoRA 4-bit causait le JSON malformé (parse 81 %) ; bf16 → 100 %.
2. **r=32 aide le petit modèle, pas le gros** : 1.7B +6 pts (63 → 68,9 %) ; 4B −2 pts
   (76,3 → 74,1 %, dans le bruit). Le 4B était déjà à l'aise en r=16.
3. **Le DPO ne change pas les métriques gold** (identiques au SFT) — il n'apporte que la
   qualité de forme (politesse/explication), non mesurée ici.
4. **Non-déterminisme** : ±2-3 pts d'exactitude, ±1 cas urgent selon le seed → les écarts
   < 3 pts ne sont pas significatifs.

## Modèles finaux à garder

| Modèle | Parse | Exactitude | Binaire | Rôle |
|---|---|---|---|---|
| **1.7B `lora_fiche_bf16_r32`** (+ `lora_dpo_v3`) | 100 % | 68,9 % | **4,5 %** | le modèle du brief, sûr |
| **4B `qwen35_fiche_stage2`** (+ `qwen35_dpo_v3`) | 100 % | **76,3 %** | 10,6 % | le plus exact |
| 1.7B `lora_stage2_v2` / 4B `qwen35_stage2` | — | — | — | référence ancien format |

Le 1.7B (exigé) est récupéré et sûr (4,5 % de sous-triage urgent) ; le 4B reste le plus
exact (76,3 %). Le choix final dépend du critère : **sécurité** (1.7B) ou **exactitude** (4B).
