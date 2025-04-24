@echo off
echo [INFO] Configuration du PYTHONPATH...
set PYTHONPATH=%cd%

echo [INFO] Lancement de l'API FastAPI sans --reload (mode stable Windows)...
uvicorn api.main:app --host 127.0.0.1 --port 8000

pause

