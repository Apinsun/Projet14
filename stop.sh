#!/usr/bin/env bash
# Arrête l'agent de triage.
#   - option A (start.sh) : conteneur app + vLLM sur l'hôte ;
#   - option B (docker-compose) : services compose.
set -euo pipefail
cd "$(dirname "$0")"

stopped=0

# Option B : services docker-compose éventuels
if command -v docker >/dev/null 2>&1; then
  services="$(docker compose ps --services 2>/dev/null || true)"
  if [ -n "$services" ]; then
    echo "⏹️  Arrêt des services docker-compose…"
    docker compose down
    stopped=1
  fi
fi

# Option A : conteneur app lancé par start.sh
if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' | grep -q '^triage-app$'; then
  echo "⏹️  Arrêt du conteneur app…"
  docker rm -f triage-app >/dev/null
  stopped=1
fi

# vLLM sur l'hôte (PID + filet pkill, y compris enfants orphelins)
if [ -f /tmp/vllm.pid ]; then
  pid="$(cat /tmp/vllm.pid 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "⏹️  Arrêt de vLLM (PID $pid)…"
    kill "$pid" 2>/dev/null || true
    stopped=1
  fi
  rm -f /tmp/vllm.pid
fi
pkill -f "venv-vllm" 2>/dev/null && stopped=1 || true
sleep 2
pkill -9 -f "venv-vllm" 2>/dev/null || true

if [ "$stopped" -eq 0 ]; then
  echo "ℹ️  Rien à arrêter."
else
  echo "✅ Arrêt effectué."
fi
