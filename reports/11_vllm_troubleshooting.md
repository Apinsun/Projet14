# Dépannage vLLM — problèmes rencontrés et leçons pour la semaine 4

**Contexte** : installation et premier lancement de vLLM (v0.29.0) en local pour la
baseline (voir `reports/10_baseline_eval.md`).
**Objectif** : consigner les obstacles pour ne pas les revivre au moment du déploiement
Docker (semaine 4).

---

## 1. Problèmes rencontrés (et fixes)

### 1.1 `FileNotFoundError: 'ninja'`

- **Symptôme** : au premier lancement, l'engine core échoue pendant le JIT de flashinfer.
- **Cause** : flashinfer compile ses noyaux CUDA à la volée via l'outil de build `ninja`,
  absent du système.
- **Fix** : `uv pip install --python .venv-vllm/bin/python ninja`.
- **Leçon** : l'image Docker officielle `vllm/vllm-openai` inclut déjà ninja → non-repro
  en prod.

### 1.2 `nvcc fatal: Unknown option '--compress-mode=size'`

- **Symptôme** : après l'installation de ninja, le build flashinfer échoue à la
  compilation des noyaux de sampling.
- **Cause** : **mismatch CUDA**. Le `nvcc` système est en **CUDA 12.4**, alors que
  vLLM 0.29 + torch 2.13 sont en **CUDA 13.0**. L'option `--compress-mode=size` n'existe
  qu'à partir de nvcc 12.8. flashinfer appelle `/usr/bin/nvcc` (système), donc échoue.
- **Fix appliqué** : `VLLM_USE_FLASHINFER_SAMPLER=0` → vLLM retombe sur le sampler
  PyTorch natif (aucun JIT). Suffisant pour la baseline.
- **Fix alternatifs** (non retenus) :
  - installer un nvcc ≥ 12.8 (`nvidia-cuda-nvcc-cu12`/`cu13` via pip, ou un toolkit CUDA) ;
  - utiliser une image Docker avec CUDA cohérent.
- **Leçon semaine 4** : l'image `vllm/vllm-openai` embarque un nvcc cohérent avec sa
  version CUDA → le problème disparaît en Docker. En local, préférer le sampler natif
  ou aligner le nvcc.

### 1.3 Premier démarrage long (compilation)

- **Symptôme** : ~6 min de démarrage au premier lancement (capture CUDA graphs +
  compilation inductor pour `max_model_len=40960` + 51 tailles de cudagraphs).
- **Fix** : `--enforce-eager` (désactive la compilation) + `--max-model-len 8192`
  (on n'a pas besoin de 40k tokens pour le triage).
- **Leçon** : pour itérer en dev, `--enforce-eager` ; en prod, accepter le premier
  démarrage lent (compilation amortie ensuite).

---

## 2. Observations utiles

### 2.1 VRAM réservée par défaut

vLLM réserve par défaut **92 % de la VRAM** (`--gpu-memory-utilization 0.92`) pour le
cache KV, même sans requête. Sur une RTX 3090 24 Go, on a observé ~23,7 Go occupés au
repos.

- Pour cohabiter avec d'autres processus GPU (ex. Ollama/Leila pour la génération) :
  `--gpu-memory-utilization 0.5` (ou `--kv-cache-memory <octets>`).
- Pour un modèle 1.7B, 0.5 suffit largement (le modèle fait ~3,4 Go).

### 2.2 Concurrence non exploitée (pour l'instant)

Le harness d'évaluation envoie les requêtes **séquentiellement** (une à la fois). Le
gain de vLLM (continuous batching) n'est donc **pas** exploité pendant l'évaluation —
ce n'est pas gênant pour une baseline, mais un vrai test de charge en prod devra
envoyer des requêtes **concurrentes** pour mesurer le débit réel.

---

## 3. Checklist semaine 4 (déploiement Docker)

1. ✅ Image officielle `vllm/vllm-openai` (ninja + nvcc cohérents inclus).
2. ✅ **Fusionner le LoRA** avant serving (`save_pretrained_merged`) → pas de `--enable-lora`.
3. ✅ `--gpu-memory-utilization` adapté à l'host (et pas 0.92 par défaut si cohabitation).
4. ✅ `--max-model-len` borné (8192 suffit pour le triage).
5. ✅ Épingler la **version vLLM** (les comportements LoRA/Qwen sont version-sensibles).
6. ✅ Passerelle FastAPI devant vLLM (logique de triage + Pydantic + audit), pas vLLM exposé nu.
7. ⚠️ Tester la **concurrence** (plusieurs patients simultanés) pour valider le débit.
