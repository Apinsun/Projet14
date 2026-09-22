# 17 — Plan d'adaptation SFT pour Qwen3.5-4B

**Date** : 22/09/2026 · **Statut** : 🟡 (modèle téléchargé, script à adapter)

## Objectif

Tester notre « meilleure recette » (SFT LoRA + multi-tours) sur un modèle de base plus
gros et plus récent : **Qwen3.5-4B**, pour montrer que le gain se maintient avec un
modèle plus intelligent.

## Modèle

| Champ | Valeur |
|---|---|
| Dépôt | `Qwen/Qwen3.5-4B` (**Instruct**, PAS `-Base` qui est pré-entraîné seul) |
| Type | **VLM** (vision + langage) — utilisé en **text-only** pour le triage |
| Params | 4B · 32 couches · hidden 2560 · **hybride Gated DeltaNet + Gated Attention** |
| Contexte | 262 144 tokens natifs |
| Thinking | `<think>...</think>` **identique à Qwen3** ✓ (compatible avec notre format) |
| Fichiers | 2 shards safetensors + config + tokenizer + chat_template |

## Changements obligatoires vs notre recette Qwen3-1.7B

| Paramètre | Qwen3-1.7B (actuel) | **Qwen3.5-4B** | Pourquoi |
|---|---|---|---|
| Quantification | QLoRA 4-bit | **bf16 LoRA** (`load_in_16bit=True`) | Unsloth : « QLoRA not recommended for Qwen3.5 » |
| `target_modules` | `q,k,v,o,gate,up,down` | **`"all-linear"`** | architecture hybride DeltaNet (noms de couches différents) |
| `transformers` | v4 | **v5** | Qwen3.5 exige transformers v5 |
| VRAM entraînement | ~5 Go | **~10 Go** (bf16) | rentre dans 24 Go |
| Sampling | temp 0.6 | **temp 1.0 + `presence_penalty=1.5`** | recommandé Qwen3.5 (anti-répétition) |

**Inchangé** : r=16, alpha=16, dropout=0, LR 2e-4 cosine (à confirmer), batch effectif 8,
max_seq_length 2048, stratégie 2 étapes (reprise base), données (4 500 base + 1 374 triage).

## Serving vLLM (semaine 4)

- vLLM récent requis (notre v0.29 devrait convenir) + `--reasoning-parser qwen3`.
- `--language-model-only` pour ignorer l'encodeur vision (gain de VRAM).
- `max-model-len` : on peut rester à 8 192 (dialogues courts), pas besoin des 262 K.

## À faire

1. Mettre à jour `.venv-train` : `transformers v5` (+ vérifier Unsloth à jour).
2. Adapter `scripts/train_sft.py` : `load_in_16bit=True`, `target_modules="all-linear"`,
   nom de modèle `Qwen/Qwen3.5-4B`.
3. Télécharger/entraîner la base médicale (stage1) sur 4B, puis le triage (stage2).
4. Évaluer (gold FR zero-shot + interactif) et comparer au 1.7B.

## Risque / point de vigilance

- **Mise à jour transformers v5** : peut casser l'entraînement Qwen3-1.7B existant dans
  le même venv. À faire proprement (ou venv séparé).
- **QLoRA → bf16** : le 4B en bf16 LoRA tient en 10 Go, mais le modèle de base complet
  (4B bf16 ≈ 8 Go) + adaptateurs + grads = ~10-12 Go. OK sur 24 Go.
