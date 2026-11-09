#!/usr/bin/env bash
# Lance l'API en mode DEV : bind mount du dossier courant + hot reload.
# À exécuter depuis le dossier fastapi_demo_aifs02/
#   bash scripts/run_dev_mac_linux.sh
set -e
docker run -it --rm \
  --name api-aifs02 \
  -p 8001:8000 \
  --env-file .env \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)":/app \
  api-aifs02 \
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
