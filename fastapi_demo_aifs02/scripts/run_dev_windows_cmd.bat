@echo off
REM Lance l'API en mode DEV : bind mount du dossier courant + hot reload.
REM À exécuter dans l'invite de commandes (cmd) depuis le dossier fastapi_demo_aifs02\
REM   scripts\run_dev_windows_cmd.bat
docker run -it --rm ^
  --name api-aifs02 ^
  -p 8001:8000 ^
  --env-file .env ^
  -v "%cd%:/app" ^
  api-aifs02 ^
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
