# 🚀 Déploiement en production du projet P08 - Segmentation sémantique

Ce document décrit le processus complet de déploiement en production de l'API FastAPI et de l'interface Gradio pour le projet P08 (segmentation d’images urbaines - Cityscapes).

---

## 🧱 Infrastructure cible : AWS EC2 (Ubuntu 24.04)

### 🔧 Spécifications utilisées :
- **Instance** : t2.micro (gratuite)
- **OS** : Ubuntu Server 24.04 LTS
- **Ports ouverts** : 22 (SSH), 8000 (API FastAPI), 7860 (Gradio)

---

## 📦 Contenu déployé sur EC2

### Structure du projet `/home/ubuntu/P08_api_minimal` :
```
P08_api_minimal/
├── api/                       # Code FastAPI
├── app/                       # Interface Gradio
├── models/                    # Modèle UNet Mini (.h5)
├── outputs/                   # Logs et résultats batch
├── src/                       # Modules utiles à l'inférence (utils, metrics)
├── venv_gradio/               # Environnement virtuel Python (Gradio)
├── launch_gradio.sh           # Script de lancement Gradio
├── launch_api.sh              # Script de lancement Docker FastAPI
├── Dockerfile                 # Image Docker de l'API
├── requirements.txt
```

---

## 🚀 Déploiement de l’API FastAPI

### Dockerfile de l'API :
Contient une application FastAPI servie par `uvicorn`, qui :
- charge un modèle `.h5` depuis `models/`
- applique le traitement d’image
- renvoie un masque colorisé encodé en base64 via l’endpoint `/predict`

### Lancement manuel :
```bash
docker build -t fastapi-p8 .
docker run -d -p 8000:8000 --name fastapi-p8-instance fastapi-p8
```

### Lancement automatique au redémarrage EC2 :
Crontab (accessible via `crontab -e`) :
```
@reboot /home/ubuntu/P08_api_minimal/launch_api.sh >> /home/ubuntu/api.log 2>&1
```

---

## 🎨 Déploiement de l’interface Gradio

### Gradio UI :
L'interface `gradio_ui.py` affiche :
- image originale, masque prédit, superposition
- légende dynamique, log des inférences, batch test

### Lancement manuel :
```bash
source venv_gradio/bin/activate
python app/gradio_ui.py
```

### Lancement automatique :
Crontab :
```
@reboot /home/ubuntu/P08_api_minimal/launch_gradio.sh >> /home/ubuntu/gradio.log 2>&1
```

---

## ✅ Résultat en production

- Interface Swagger (API) : http://<IP_EC2>:8000/docs
- Interface Gradio : http://<IP_EC2>:7860

---

## 🧪 Vérification post-reboot

Après chaque redémarrage EC2 :
```bash
docker ps                   # Vérifie que l’API tourne
curl localhost:7860         # Vérifie que Gradio est actif
```

---

## ✨ Bonus possible

- Déploiement de Gradio sur Hugging Face Spaces (version publique)
- Ajout de monitoring, alertes ou nom de domaine
- Script de packaging `.zip` pour OpenClassrooms

---

© Projet P08 — OpenClassrooms — Arnaud