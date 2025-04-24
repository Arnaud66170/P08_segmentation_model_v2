@echo off
echo ========================================
echo 🧠 LANCEMENT UVICORN EN MODE DEBUG
echo ========================================
echo.
set PYTHONPATH=%CD%
uvicorn api.main:app --reload --log-level debug


pause