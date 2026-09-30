#!/usr/bin/env bash
# Arrête l'agent de triage : conteneurs Docker (start.sh) et/ou processus locaux
# vLLM + FastAPI lancés en arrière-plan.
set -euo pipefail
cd "$(dirname "$0")"

stopped_any=0

# 1. Conteneurs Docker (si lancés via start.sh)
if command -v docker >/dev/null 2>&1; then
  services="$(docker compose ps --services 2>/dev/null || true)"
  if [ -n "$services" ]; then
    echo "⏹️  Arrêt des conteneurs Docker…"
    docker compose down
    stopped_any=1
  fi
fi

# 2. Processus locaux (vLLM + uvicorn) démarrés en arrière-plan
for pidfile in /tmp/vllm.pid /tmp/app.pid; do
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "⏹️  Arrêt du processus $pid ($(basename "$pidfile" .pid))…"
      kill "$pid" 2>/dev/null || true
      stopped_any=1
    fi
    rm -f "$pidfile"
  fi
done

if [ "$stopped_any" -eq 0 ]; then
  echo "ℹ️  Rien à arrêter (ni conteneur, ni processus local)."
else
  echo "✅ Arrêt effectué."
fi
