# 🧠 Segmentation d'Images Urbaines - UNet Mini (Projet P08)

Bienvenue sur ce Space Hugging Face dédié à la segmentation sémantique d’images issues du dataset **Cityscapes**, dans le cadre du projet P08 (OpenClassrooms - Ingénieur IA).

Ce projet applique un modèle **UNet Mini** entraîné pour segmenter les scènes urbaines en 8 grandes classes regroupées.

---

## 🖼️ Fonctionnalités de l'interface

- **Upload d'image** (format `.png` recommandé)
- **Masque prédit** par le modèle (depuis une API FastAPI hébergée sur EC2)
- **Superposition semi-transparente** de l'image et du masque
- **Affichage du temps d'inférence**
- Interface optimisée pour les démonstrations et la soutenance

---

## 🔁 Fonctionnement

Ce Space envoie les images à une API FastAPI (hébergée sur AWS EC2) via un appel `POST` à l’endpoint `/predict`, puis décode et affiche le masque de segmentation retourné (encodé en `base64` au format PNG).

---

## 🌐 API utilisée

L'API de prédiction est hébergée ici :  
```
http://3.83.179.62:8000/docs
```

Endpoint utilisé : `POST /predict` avec fichier image `.png`

---

## 🔧 Technologies utilisées

- **Gradio** pour l'interface utilisateur interactive
- **FastAPI** pour l'inférence en ligne
- **Docker** pour le déploiement de l’API sur EC2
- **UNet** pour la segmentation sémantique (TensorFlow 2.10.1)

---

## 👨‍💻 Auteur

**Arnaud (arnaud66170)**  
Projet réalisé dans le cadre du parcours **Ingénieur IA – OpenClassrooms**