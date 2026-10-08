#!/usr/bin/env bash
# Démarre l'agent de triage — « option A » (seule l'app est conteneurisée) :
#   - vLLM tourne sur l'hôte (venv .venv-vllm), port 8000 ;
#   - l'app FastAPI est conteneurisée (image GHCR), port 8080, et joint vLLM
#     via host.docker.internal.
#
# Le brief demande de conteneuriser l'application ; vLLM est le moteur d'inférence
# appelé par l'app. (Option B = tout en conteneurs : voir docker-compose.yml.)
#
# Usage : ./start.sh [chemin-du-modele]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-models/lora_dpo_v3_merged}"
APP_IMAGE="${TRIAGE_APP_IMAGE:-ghcr.io/apinsun/projet14:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker n'est pas installé." >&2
  exit 1
fi

# --- 1. vLLM sur l'hôte ---
if curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "ℹ️  vLLM déjà en route."
else
  echo "🚀 Démarrage de vLLM sur l'hôte (modèle : $MODEL)…"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup .venv-vllm/bin/vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
    --enforce-eager --max-model-len 8192 > /tmp/vllm_serve.log 2>&1 &
  echo $! > /tmp/vllm.pid
  for _ in $(seq 1 90); do
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "✅ vLLM prêt."; break; }
    sleep 2
  done
  curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 \
    || { echo "❌ vLLM n'a pas démarré (voir /tmp/vllm_serve.log)." >&2; exit 1; }
fi

# --- 2. App conteneurisée (image GHCR) ---
if docker ps --format '{{.Names}}' | grep -q '^triage-app$'; then
  echo "ℹ️  Conteneur app déjà lancé."
else
  echo "🚀 Récupération de l'image + démarrage du conteneur app…"
  docker pull "$APP_IMAGE" >/dev/null
  docker rm -f triage-app >/dev/null 2>&1 || true
  docker run -d --name triage-app -p 8080:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e TRIAGE_VLLM_URL=http://host.docker.internal:8000/v1 \
    -e TRIAGE_VLLM_MODEL="$MODEL" \
    -e TRIAGE_AUDIT_PATH=/app/logs/audit.jsonl \
    -v "$PWD/logs:/app/logs" \
    "$APP_IMAGE"
  for _ in $(seq 1 15); do
    curl -s -m 2 http://localhost:8080/health >/dev/null 2>&1 && { echo "✅ app prête."; break; }
    sleep 1
  done
fi

echo ""
echo "🩺 Interface web : http://localhost:8080"
echo "📚 Documentation API : http://localhost:8080/docs"
echo ""
echo "Logs app : docker logs -f triage-app"
echo "Arrêter  : ./stop.sh"
