# Plan SFT (LoRA) — Qwen3-1.7B avec Unsloth

**Objet** : plan détaillé de l'étape de Supervised Fine-Tuning (semaine 2) : déroulement,
dépendances, paramètres cruciaux, et best practices issues de la recherche.
**Date** : 2026-09-14

---

## 1. Déroulement de l'étape

1. **Environnement d'entraînement séparé** (`.venv-train`, Python 3.12) — ne pas polluer
   le venv poetry ni `.venv-vllm` (Unsloth a ses propres versions torch/CUDA, potentiellement
   incompatibles avec vLLM).
2. **Vérifier le format des données** : nos jeux sont déjà en `{"messages": [...]}`,
   directement compatibles `tokenizer.apply_chat_template` + `SFTTrainer`.
3. **Fusionner les jeux** : `sft_train` (5 000 : MedQuAD/FrenchMedMCQA/MediQA) + triage
   synthétique (600) → ~5 600 exemples.
4. **Pilot run** (100–200 exemples, 1 epoch) → valider le pipeline (loss, mémoire, format).
5. **Run complet** (5 600 exemples, 1–2 epochs) + checkpoint.
6. **Évaluer** : `sft_val` (500) + `clinical_eval` (1 038) + jeux gold via vLLM (harness existant).
7. **Itérer** sur hyperparamètres si besoin.
8. **Fusionner le LoRA** (`save_pretrained_merged`) → servir via vLLM, relancer le harness.

---

## 2. Dépendances

| Paquet | Rôle |
|---|---|
| `unsloth` + `unsloth_zoo` | framework d'entraînement accéléré (SFT/DPO), patche transformers/trl |
| `torch` (+ build CUDA) | backend (version alignée avec Unsloth, ex. cu124/cu126) |
| `transformers` | chargement du modèle/tokenizer |
| `trl` | `SFTTrainer`, `DPOTrainer` |
| `peft` | LoRA |
| `datasets` | chargement des jeux (déjà présent) |
| `accelerate` | distribution |
| `bitsandbytes` | quantification 4-bit (QLoRA) |
| `xformers` *(optionnel)* | accélération attention |

> ⚠️ **CUDA** : comme pour vLLM, aligner torch/CUDA. Unsloth publie des wheels par
> version CUDA (cu121, cu124, cu126…) ; RTX 3090 = sm_86, compatible partout.

---

## 3. Paramètres cruciaux (recommandations)

| Paramètre | Valeur recommandée | Pourquoi |
|---|---|---|
| `max_seq_length` | **2048** | nos dialogues + fiches sont courts ; 2048 suffit (Qwen3 = 40960 natif) |
| `load_in_4bit` | **True** (QLoRA) | 1.7B tiendrait en 8-bit, mais 4-bit = plus rapide + standard Unsloth |
| LoRA `r` | **16** (essayer 8) | r=16 défaut robuste ; r=8 plus prudent pour tâche étroite/petit dataset |
| LoRA `alpha` | **16** (= r) ou 32 (= 2r) | scaling du signal LoRA |
| LoRA `dropout` | **0** | recommandé Unsloth |
| `target_modules` | `q,k,v,o,gate,up,down_proj` | projections attention + MLP (toutes les couches linéaires) |
| learning rate | **2e-4** (pic) | standard LoRA ; plage 1e-4 → 3e-4 ; trop bas (2e-5) = sous-apprentissage |
| scheduler | **cosine** + warmup ~5 % | décroissance douce |
| optimizer | **adamw_8bit** | défaut Unsloth (moins de VRAM) |
| epochs | **1–2** | ~5 600 exemples : 1 epoch suffit souvent, 2 si sous-apprentissage |
| batch size | per-device **2–4** + grad_accum 2–4 (effectif ~8–16) | stabilité + fit VRAM |
| `use_rslora` / `random_state` | seed fixe (42) | reproductibilité |

---

## 4. Ce que dit la recherche (Qwen3 + best practices)

1. **Taille de dataset** : 1 000–5 000 exemples de qualité « suffisent » pour du SFT
   (instruction following). Nos ~5 600 sont dans la bonne fourchette.
2. **Risque de sur-apprentissage** : plus le dataset est petit, plus le risque monte.
   → r bas (8–16), 1–2 epochs, validation sur `sft_val`.
3. **Learning rate** : le plus important à bien régler. 2e-4 est le point de départ LoRA
   courant (certains montent à 3e-4 avec `adam_beta2=0.95`, `weight_decay=0.1`). Une
   étude (« lr-matters-lora ») souligne que beaucoup de travaux règlent mal le LR.
4. **Rang LoRA** : un retour communautaire (Qwen3-8B) note que **r=8 préservait mieux le
   comportement** (`/think`, `/no_think`), r=32+ le dégradait, r=64 « cassait » la sortie,
   r=128 « sur-apprenait ». → rester bas (8–16) pour une tâche étroite.
5. **Mix raisonnement / non-raisonnement** : Unsloth recommande **75 % raisonnement /
   25 % non-raisonnement** pour préserver le raisonnement de Qwen3. Nos données ont un
   mix naturel (dialogues avec `<think>` = raisonnement ; Q&A de base = non).
6. **⚠️ Mode « thinking » natif de Qwen3** : Qwen3 a un mode `<think>...</think>` natif
   (activé par défaut). Notre format utilise `<think>...</think>`. Décision à
   trancher (cf. §5).

---

## 5. Points de vigilance spécifiques

### 5.1 `<think>` natif (décision)

**Décision : utiliser les balises natives Qwen3 `<think>...</think>`** (renommer nos
`<think>`).

- Qwen3 a un mode thinking natif (`<think>...</think>`, contrôlé par `enable_thinking`).
- **Avantages** : aligné sur le pré-entraînement (apprentissage plus robuste) ;
  `reasoning_parser` vLLM qui sépare automatiquement le raisonnement de la réponse
  (≈ notre besoin « masquer au patient, garder pour audit ») ; mode natif à l'inférence.
- **Inconvénients** : refactor trivial (find/replace + régénérer) ; risque de CoT verbeux
  (mitigé par le SFT qui impose la structure concise).
- La **structure interne** de nos blocs (faits, red flags, plage, question) est conservée,
  seul l'habillage `<think>` → `<think>` change.

### 5.2 Le triage risque d'être noyé

600 exemples de triage sur ~5 600 = **11 %**. Le comportement de triage (fiche, prudence,
ton patient) risque d'être sous-représenté face aux 5 000 Q&A médicaux génériques.
- **Décision** : **sur-échantillonner le triage (×2–3 → 20–30 %)** en répétant les
  600 exemples dans le mix d'entraînement.

### 5.3 Ratio FR/EN

Objectif 50/50 respecté sur la base (2 400 FR / 2 600 EN). Le triage est majoritairement
FR → à vérifier après mixage pour ne pas déséquilibrer.

---

## 6. Plan de validation

| Étape | Mesure |
|---|---|
| **Pilot run** (≈200 ex., 1 epoch) | loss décroissante (2,x → 0,x) + **parse ≈ 100 %** sur 50 vignettes synthétiques hors entraînement + parse gold (few-shot) qui saute de 53 % → > 80 % (format appris) |
| Après SFT | parse `[FICHE]` ≈ 100 % sur `sft_val` + `clinical_eval` |
| Jeux gold (vLLM) | exactitude 3 classes, **sous-triage ≈ 0**, comparaison vs baseline (0/87) |
| Revue humaine | échantillon de dialogues (comme convenu) |

**Critère de succès POC** : parse ~100 % + sous-triage binaire ≈ 0 sur Ramaswamy/Levine,
et un delta net vs la baseline (0/87 → cible > 80 % d'exactitude 3 classes).

---

## 7. Résultats du pilot run (200 vignettes, 1 epoch)

| Métrique | Base (few-shot) | Pilot SFT | Verdict |
|---|---|---|---|
| Parse `[FICHE]` | 53 % | **64 %** | ✅ format appris |
| Ramaswamy parse | 0 % | **22 %** | ✅ il ne parseait jamais |
| Exactitude | 33 % | **52 %** | ✅ améliorée |
| Sous-triage | 8,7 % | **30 %** | ⚠️ dégradé (attendu) |

**Conclusion** : le pilot run **valide le pipeline** (loss 2.76 → 1.5 en 25 steps) et
montre que le **format s'apprend vite**. Mais 200 exemples ne suffisent **pas pour la
sécurité** (sous-triage 30 %, 5 urgences `Emergent` ratées) → le run complet
(5 600 ex., triage ×3) + le DPO sont nécessaires.

### 7.1 Pièges techniques rencontrés (à retenir)

1. **TRL 0.24** : API modifiée — `processing_class` (au lieu de `tokenizer`),
   `SFTConfig.max_length` (au lieu de `max_seq_length`), `dataset_text_field` dans
   `SFTConfig`. Unsloth attend `formatting_func -> list[str]`.
2. **cuDNN** : `Invalid handle. Cannot load symbol cudnnGetVersion` → fixer
   `LD_LIBRARY_PATH` vers les libs nvidia du venv (`scripts/train.sh` le fait).
3. **Import order** : `from unsloth import FastModel` avant trl/transformers.
