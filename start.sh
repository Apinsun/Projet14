#!/usr/bin/env bash
# Lance l'agent de triage (vLLM + FastAPI) avec Docker Compose.
# Prérequis : Docker + nvidia-container-toolkit, modèle fusionné dans
#   models/lora_dpo_v3_merged (monté en volume par docker-compose.yml).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker n'est pas installé." >&2
  exit 1
fi

echo "🚀 Démarrage des conteneurs (vLLM + FastAPI)…"
docker compose up --build -d

echo ""
echo "🩺 Interface web : http://localhost:8080"
echo "📚 Documentation API : http://localhost:8080/docs"
echo ""
echo "Suivre les logs : docker compose logs -f app"
echo "Arrêter          : docker compose down"
