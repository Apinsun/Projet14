#!/usr/bin/env bash
# Lance l'entraînement SFT avec les libs nvidia du venv dans LD_LIBRARY_PATH.
# (sinon : "Invalid handle. Cannot load symbol cudnnGetVersion")
set -euo pipefail
cd "$(dirname "$0")/.."
NVLIBS="$(ls -d .venv-train/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
exec .venv-train/bin/python scripts/train_sft.py "$@"
