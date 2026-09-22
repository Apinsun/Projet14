# 15 — Plan SFT v2 (multi-tour) — Qwen3-1.7B

**Objet** : plan précis de la nouvelle itération SFT, qui intègre le dataset multi-tours
(rapport 14) pour corriger le mode interactif.
**Date** : 21/09/2026

---

## 1. Données utilisées

| Jeu | Fichier | Exemples | Rôle |
|---|---|---|---|
| Base médicale | `data/processed/final/sft_train.jsonl` | 4 500 | **Stage1** — connaissance médicale (Q&A MedQuAD/FrenchMedMCQA/MediQA) |
| Vignettes mono-tour | `data/processed/triage/sft_vignettes.jsonl` | 300 | **Stage2** — symptôme complet → fiche (comportement mono-tour, éval gold) |
| Multi-tours | `data/processed/triage/sft_multiturn_full.jsonl` | 1 074 | **Stage2** — symptôme → questions → fiche (comportement interactif) |
| ~~Dialogues scriptés~~ | `sft_dialogues_dressed.jsonl` | ~~300~~ | ❌ **abandonnés** (ouverture vague, le bug qu'on corrige) |

**Total Stage2 = 1 374 exemples** (300 vignettes + 1 074 multi-tours, ratio 22/78).

> Points vérifiés :
> - les 300 vignettes ne contiennent **aucun** symptôme à la 3e personne (déjà reformulées en « je ») ;
> - le format des fiches est **`<FICHE>...</FICHE>`** (plus `[FICHE]`, cf. drift rapport 13) ;
> - l'adaptateur **`models/lora_stage1`** (base 4 500) existe → **réutilisé**, pas re-entraîné.

---

## 2. Stratégie (2 étapes, comme la recette éprouvée du rapport 13)

| Étape | Données | Epochs | Objectif |
|---|---|---|---|
| **Stage1** | base 4 500 | 1 (déjà fait) | connaissance médicale → `--resume-from models/lora_stage1` |
| **Stage2** | triage 1 374 | **2** | format `<FICHE>` + questionnement + prudence |
| *Stage3 (option)* | triage 1 374 | +1 (LR 5e-5) | « cool down » si sous-apprentissage |

**Pourquoi pas de mix unique** : le run « full » (base + triage mélangés) avait dilué le
format à 25 % de parse (rapport 13 §3.1). On reste en 2 étapes.

---

## 3. Hyperparamètres (recette éprouvée, inchangée)

| Paramètre | Valeur |
|---|---|
| LoRA `r` | **16** |
| `lora_alpha` | **16** (= r) |
| `lora_dropout` | **0** |
| `target_modules` | `q,k,v,o,gate,up,down_proj` |
| Quantification | **QLoRA 4-bit** (`load_in_4bit=True`) |
| Learning rate | **2e-4** (stage1/2) · **5e-5** (stage3) |
| Scheduler | **cosine**, warmup 5 steps |
| Optimizer | **adamw_8bit**, weight_decay 0.01 |
| Batch | per-device **2** × grad_accum **4** = **effectif 8** |
| `max_seq_length` | **2048** |
| Précision | **bf16** (RTX 3090) |
| Seed | **42** |

---

## 4. Nombre de steps

Avec batch effectif 8 :

| Étape | Exemples | Steps/epoch | Total |
|---|---|---|---|
| Stage1 (base) | 4 500 | 4 500 ÷ 8 = 563 | réutilisé |
| Stage2 (triage) | 1 374 | 1 374 ÷ 8 = **172** | **344** (2 epochs) |
| Stage3 (option) | 1 374 | 172 | +172 |

→ **~344 steps pour le Stage2**, ~4× plus de matière triage que l'itération précédente
(1 374 vs 600), ce qui devrait mieux ancrer le questionnement.

---

## 5. Commande

```bash
# Stage2 (reprise de stage1)
./scripts/train.sh \
  --data triage \
  --resume-from models/lora_stage1 \
  --epochs 2 \
  --output-dir models/lora_stage2_v2 \
  --merge

# Stage3 optionnel (cool down)
./scripts/train.sh \
  --data triage \
  --resume-from models/lora_stage2_v2 \
  --epochs 1 \
  --lr 5e-5 \
  --output-dir models/lora_stage3_v2 \
  --merge
```

> ⚠️ **À faire avant** : `scripts/train_sft.py` charge encore `sft_dialogues_dressed.jsonl`
> dans `--data triage`. Remplacer ce fichier par `sft_multiturn_full.jsonl`.

---

## 6. Évaluation

| Ce qu'on mesure | Comment |
|---|---|
| **Mono-tour** (parse, exactitude, sous-triage) | harness gold FR, **zero-shot** (le few-shot dégrade après SFT) |
| **Mode interactif** (questionnement) | 🆕 à construire : le modèle pose-t-il des questions avant la fiche ? (mesure du bug corrigé) |
| Parse `<FICHE>` | doit rester ≈ 95 % |

**Critère de succès** : parse ≈ 95 % conservé, sous-triage ≈ 12 % (cible DPO ensuite),
et surtout **le modèle questionne avant de conclure** en interactif.

---

## 7. Points de vigilance

1. **Mettre à jour `train_sft.py`** : `--data triage` → vignettes + `sft_multiturn_full.jsonl`
   (plus `sft_dialogues_dressed.jsonl`).
2. **Format `<FICHE>`** : déjà cohérent dans les données (parser tolérant pour l'éval).
3. **Zero-shot uniquement** (pas de few-shot).
4. **Surveiller la loss** : décroissance attendue 2.x → ~0.5-1 en Stage2.
5. **VRAM** : QLoRA 4-bit sur 1.7B ≈ 5-8 Go (pas de contrainte sur 24 Go).
