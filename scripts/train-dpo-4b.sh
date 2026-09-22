#!/usr/bin/env bash
# Lance le DPO sur Qwen3.5-4B (venv .venv-train-4b, bf16 LoRA + all-linear).
# Usage : ./scripts/train-dpo-4b.sh --data ... --resume-from models/qwen35_stage2 ...
set -euo pipefail
cd "$(dirname "$0")/.."
NVLIBS="$(ls -d .venv-train-4b/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
exec .venv-train-4b/bin/python scripts/train_dpo.py --bf16 --target-all-linear "$@"
