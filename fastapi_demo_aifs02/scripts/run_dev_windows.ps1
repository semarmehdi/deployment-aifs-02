# Lance l'API en mode DEV : bind mount du dossier courant + hot reload.
# À exécuter dans PowerShell depuis le dossier fastapi_demo_aifs02\
#   .\scripts\run_dev_windows.ps1
docker run -it --rm `
  --name api-aifs02 `
  -p 8001:8000 `
  --env-file .env `
  -v "${PWD}:/app" `
  api-aifs02 `
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
