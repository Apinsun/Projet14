# Baseline d'évaluation (Qwen3-1.7B de base, avant fine-tune)

**Objet** : installer vLLM, servir Qwen3-1.7B de base, et mesurer la performance de
triage **avant** tout fine-tune.
**Date** : 2026-09-14

---

## 1. Installation de vLLM

- **Version** : vLLM 0.29.0 (torch 2.13.0+cu130).
- **Environnement** : venv dédié `.venv-vllm` (Python 3.12 via `uv`), séparé du venv
  poetry — le harness et vLLM communiquent par HTTP.
- **GPU** : RTX 3090 24 Go.

### Problèmes rencontrés et corrigés (à conserver pour la semaine 4 / Docker)

| Problème | Cause | Fix |
|---|---|---|
| `FileNotFoundError: 'ninja'` | flashinfer JIT a besoin de l'outil `ninja` | `uv pip install ninja` |
| `nvcc fatal: Unknown option '--compress-mode=size'` | nvcc système = CUDA 12.4, mais flashinfer (vLLM 0.29) exige nvcc ≥ 12.8 | `VLLM_USE_FLASHINFER_SAMPLER=0` (sampler PyTorch natif) |

> Le second point est un **mismatch CUDA** : torch est en cu130, le nvcc système en 12.4.
> Désactiver le sampler flashinfer suffit pour la baseline ; en production on utilisera
> l'image Docker officielle (nvcc cohérent) ou on installera un nvcc 12.8+.

### Commande de lancement

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
.venv-vllm/bin/vllm serve Qwen/Qwen3-1.7B --port 8000 --enforce-eager --max-model-len 8192
```

API OpenAI-compatible sur `http://localhost:8000/v1`.

---

## 2. Résultat baseline

**Jeux** : Ramaswamy (39, symptômes seuls) + Levine (48) = **87 cas**.

| Métrique | Valeur |
|---|---|
| Cas évalués | 87 |
| **Fiches `[FICHE]` parsées** | **0 / 87** |
| Exactitude | — (aucun parse) |
| Sous-triage / sur-triage | — |

**Interprétation** : le Qwen3-1.7B de base ne connaît pas notre convention `[FICHE]` —
il répond en mode « thinking » natif (`<think>...</think>`) avec un texte libre, sans
jamais émettre la fiche structurée. C'est **attendu** : le format `[FICHE]` + la
discipline de triage sont précisément ce que le SFT va lui apprendre.

> Exemple de sortie du modèle de base :
> `<think> Okay, the user is saying they have chest pain that's squeezing... </think>`
> (raisonnement libre, aucune fiche, aucun niveau de triage).

### 2.1 Comparaison few-shot et paramètres d'échantillonnage

On injecte 2 exemples (1 niveau 1, 1 niveau 5) tirés des vignettes SFT (`--few-shot`).

| Configuration | Fiches parsées | Exactitude | Observation |
|---|---|---|---|
| zero-shot, greedy, max_tokens 400 | 0 / 87 | — | n'émet jamais `[FICHE]` |
| few-shot, greedy, max_tokens 400 | 6 / 87 | 0,33 | format appris, valeurs fausses |
| **few-shot, temp 0.6/top_p 0.95/top_k 20, max_tokens 2048** | **46 / 87** | 0,33 | format bien suivi (53 %), valeurs toujours fausses |

**Analyse des 46 parses** (tous sur Levine ; Ramaswamy reste à 0) :
- priorités prédites : **3 × 35**, 5 × 7, 4 × 4 → le modèle se replie sur la valeur médiane (3) ;
- **4 sous-triages** (dont 2 graves : « Emergent » [1,3] → priorité **5**) ;
- sur-triage massif (0,59) : bénin [5,5] → 3.

> **Conclusions** :
> 1. Les **paramètres d'échantillonnage Qwen3** (pas de greedy, temp 0.6, max_tokens 2048)
>    ont fait passer le taux de parse de 6 → 46 (l'ancien greedy + 400 tokens tronquait le `<think>`).
> 2. Le few-shot apprend **le format**, pas **la justesse** : le modèle de base se replie
>    sur 3 et **sous-trie des urgences en 5** — l'erreur dangereuse que le SFT + DPO doivent éliminer.
> 3. Ramaswamy (prompts plus longs) reste à 0 parse : le `<think>` natif ramble et dépasse
>    les 2048 tokens avant la fiche → confirme la nécessité du fine-tune (concision).

---

## 3. Ce que cette baseline établit

1. **Point de départ chiffré** : 0 % de fiche parsée, 0 % de triage exploitable.
2. **Le harness fonctionne de bout en bout** sur vLLM (`--backend openai`) : envoi,
   parsing, correction, métriques.
3. **La cible du SFT est claire** : faire passer ce 0 % à un taux de parse ~100 % **et**
   un triage correct (mesuré ensuite par les 3 étages de granularité).

Après le SFT Unsloth (semaine 2), on servira le **modèle fusionné** avec la même
commande et on relancera exactement ce harness pour mesurer le delta.
