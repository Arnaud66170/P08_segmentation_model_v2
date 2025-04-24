@echo off
echo [INFO] Activation de l'environnement de test...
set PYTHONPATH=%cd%

echo [INFO] Lancement du script de test local sur les images de test_images/ ...
python scripts/test_predict_local.py

pause
