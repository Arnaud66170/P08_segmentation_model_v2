
# 🚀 Déploiement du projet P8 - Segmentation d’Images sur AWS EC2

Ce document décrit étape par étape comment déployer l'API de segmentation (FastAPI) et son interface utilisateur (Gradio) sur une instance Ubuntu 22.04 AWS EC2.

---

## 📦 Contenu du projet

```
P08_segmentation_model/
├── api/
│   ├── main.py               # API FastAPI (upload image → mask)
│   ├── model_loader.py       # Chargement modèle .h5
│   └── inference.py          # Pipeline de prédiction
├── app/
│   └── gradio_ui.py          # Interface Gradio (test visuel)
├── models/
│   └── best_model_unet.h5    # Modèle entraîné sauvegardé
├── requirements.txt
├── scripts/
│   └── deploy_ec2.sh         # Script automatique d’installation (optionnel)
```

---

## ☁️ Étapes de déploiement sur AWS EC2

### 1. 📥 Créer et configurer l’instance EC2

- Type : `t2.medium` minimum (RAM > 4 Go recommandé)
- OS : `Ubuntu 22.04 LTS`
- Groupe de sécurité : ouvrir les ports suivants :
  - `22` (SSH)
  - `8000` (FastAPI)
  - `7860` (Gradio)

---

### 2. 🔐 Se connecter à l’instance EC2

```bash
ssh -i "votre_cle.pem" ubuntu@<adresse_ip_publique>
```

---

### 3. 🛠️ Installation de l’environnement

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx -y
```

---

### 4. 📂 Copier les fichiers du projet

Depuis votre machine locale :

```bash
scp -i "votre_cle.pem" -r P08_segmentation_model/ ubuntu@<adresse_ip_publique>:~/P08_segmentation_model/
```

> ⚠️ Assurez-vous de quitter la session SSH avant d'exécuter `scp`

---

### 5. 🐍 Créer et activer l’environnement virtuel

```bash
cd ~/P08_segmentation_model
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 6. 🚀 Lancer l’API FastAPI

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

L’API est maintenant accessible à l’adresse :
```
http://<ip_publique_ec2>:8000/docs
```

---

### 7. 🧪 Tester l’API (via Swagger)

- Ouvrir dans le navigateur : `http://<ip_publique_ec2>:8000/docs`
- Tester le `POST /predict` avec une image `.png` ou `.jpg`

---

### 8. 🎛️ Lancer l’interface Gradio

```bash
cd ../app
python gradio_ui.py
```

- Interface Gradio disponible sur :  
  `http://<ip_publique_ec2>:7860/`

---

## 📼 Enregistrement de la démonstration

- Utiliser OBS ou ScreenRec pour capturer :
  - Upload d'une image
  - Affichage du masque
- Sauvegarder la vidéo `.mp4` pour la soutenance

---

## 🧪 Vérifications optionnelles

- ✅ API retourne bien un masque
- ✅ Gradio affiche les deux images (originale + masque prédite)
- ✅ Pas d’erreur TensorFlow / PIL / OpenCV
- ✅ Le port est bien ouvert sur EC2

---

## 🔁 Automatisation possible

Vous pouvez tout automatiser avec un script `scripts/deploy_ec2.sh` :

```bash
bash scripts/deploy_ec2.sh
```

---

## 📌 À venir / idées d’amélioration

- Dockerisation du projet
- CI/CD avec GitHub Actions
- Monitoring des logs (CPU, temps d'inférence, etc.)
- Ajout d’un `inference_log.csv` pour tracer les requêtes

---

© Projet P8 - OpenClassrooms | Future Vision Transport 🚗🧠
