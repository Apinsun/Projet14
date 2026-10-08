#!/usr/bin/env bash
# Lance l'agent de triage (vLLM + FastAPI) avec Docker Compose.
# L'image de l'app est tirée depuis GHCR (construite par le CI/CD) ; vLLM utilise
# l'image officielle et le modèle fusionné est monté en volume.
# Prérequis : Docker + nvidia-container-toolkit, modèle dans models/lora_dpo_v3_merged.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker n'est pas installé." >&2
  exit 1
fi

echo "🚀 Récupération de l'image depuis GHCR + démarrage…"
docker compose pull
docker compose up -d

echo ""
echo "🩺 Interface web : http://localhost:8080"
echo "📚 Documentation API : http://localhost:8080/docs"
echo ""
echo "Suivre les logs : docker compose logs -f app"
echo "Arrêter          : docker compose down"
