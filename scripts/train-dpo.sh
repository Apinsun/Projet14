#!/usr/bin/env bash
# Lance le DPO sur Qwen3-1.7B (venv .venv-train, QLoRA 4-bit).
# Usage : ./scripts/train-dpo.sh --data ... --resume-from models/lora_stage2_v2 ...
set -euo pipefail
cd "$(dirname "$0")/.."
NVLIBS="$(ls -d .venv-train/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
exec .venv-train/bin/python scripts/train_dpo.py "$@"
