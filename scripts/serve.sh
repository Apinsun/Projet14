#!/usr/bin/env bash
# Lance vLLM sur le modèle de triage fusionné (LoRA déjà fusionné dans les poids).
# Usage : ./scripts/serve.sh [chemin-du-modele]
set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL="${1:-models/lora_stage3_merged}"

echo "🚀 Serveur vLLM → http://localhost:8000 (modèle : $MODEL)"
exec .venv-vllm/bin/vllm serve "$MODEL" --port 8000 --enforce-eager --max-model-len 8192
