#!/usr/bin/env bash
# Lance l'entraînement SFT Qwen3.5-4B avec le venv dédié (.venv-train-4b).
set -euo pipefail
cd "$(dirname "$0")/.."
NVLIBS="$(ls -d .venv-train-4b/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
exec .venv-train-4b/bin/python scripts/train_sft.py "$@"
