@echo off
echo [INFO] Activation de l'environnement Gradio...
set PYTHONPATH=%cd%

echo [INFO] Lancement de l'interface Gradio (app/gradio_ui.py)...
python app/gradio_ui.py

pause
