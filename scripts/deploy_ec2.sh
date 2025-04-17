#!/bin/bash

echo "🚀 Début du déploiement du projet P8 sur EC2..."

# 1. Mise à jour des paquets
sudo apt update && sudo apt upgrade -y

# 2. Installation des dépendances système
sudo apt install -y python3-pip python3-venv nginx

# 3. Création de l’environnement virtuel Python
python3 -m venv venv
source venv/bin/activate

# 4. Upgrade de pip
pip install --upgrade pip

# 5. Installation des dépendances Python
pip install -r requirements.txt

# 6. Démarrage de l'API FastAPI avec Uvicorn (en arrière-plan)
cd api
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
cd ..

# 7. Démarrage de l’interface Gradio (en arrière-plan)
cd app
nohup python gradio_ui.py &
cd ..

echo "✅ Déploiement terminé. API dispo sur port 8000, Gradio sur port 7860."
