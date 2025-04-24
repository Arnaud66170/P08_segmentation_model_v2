@echo off
echo ================================
echo  🚀 DÉMARRAGE GLOBAL DU PROJET P8
echo ================================

echo.
echo [1/3] Lancement de l'API FastAPI...
start cmd /k run_api.bat

timeout /t 5 > nul

echo [2/3] Lancement de l'interface Gradio...
start cmd /k run_gradio.bat

timeout /t 2 > nul

echo [3/3] Lancement du test batch (15 images dans test_images/)...
timeout /t 8 > nul
start cmd /k run_test.bat

echo.
echo ✅ Tous les services sont en ligne !
echo 🌐 Accès interface Gradio : http://127.0.0.1:7860
echo 📊 API REST en ligne : http://127.0.0.1:8000/docs

pause
