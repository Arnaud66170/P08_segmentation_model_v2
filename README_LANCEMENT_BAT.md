
# 🚀 Démarrage complet du projet P08 – Segmentation d’images Cityscapes

Ce guide décrit **étape par étape** comment exécuter correctement tout le projet (API FastAPI, Interface Gradio, test par lot) à l’aide de fichiers `.bat` sur Windows.

---

## 📁 Chemin du projet (à adapter si besoin)

Assurez-vous d’avoir votre projet dans le dossier suivant (Windows) :

```
C:\Users\motar\Desktop\1-openclassrooms\AI_Engineer\1-projets\P08\P08_segmentation
```

> ⚠️ Tous les `.bat` doivent être exécutés **depuis ce dossier**.

---

## 🖥️ Terminal recommandé

Utilisez :

- ✅ **cmd.exe** (Terminal Windows classique)
- ✅ ou **Git Bash** avec la commande `cmd.exe /c run_all.bat`

> ❌ Ne pas utiliser `./run_all.bat` sous Git Bash directement (problèmes de compatibilité `start cmd /k`)

---

## 🔃 Fichiers `.bat` disponibles

| Script             | Fonction                                                                 |
|--------------------|--------------------------------------------------------------------------|
| `run_api.bat`      | Lance l’API FastAPI localement (127.0.0.1:8000)                          |
| `run_gradio.bat`   | Lance l’interface Gradio (127.0.0.1:7860)                                |
| `run_test.bat`     | Lance les prédictions sur les images du dossier `test_images/`          |
| `run_all.bat`      | Lance automatiquement les 3 étapes ci-dessus (API + Gradio + test batch) |

---

## 🌐 Accès aux services

| Composant        | URL locale à ouvrir dans le navigateur                   |
|------------------|----------------------------------------------------------|
| Interface Gradio | [http://127.0.0.1:7860](http://127.0.0.1:7860)            |
| API FastAPI      | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  |

---

## 🔥 Exemple d’exécution complète (via cmd.exe)

```bat
cd C:\Users\motar\Desktop\1-openclassrooms\AI_Engineer\1-projets\P08\P08_segmentation
run_all.bat
```

ou via **Git Bash** :

```bash
cd /c/Users/motar/Desktop/1-openclassrooms/AI_Engineer/1-projets/P08/P08_segmentation
cmd.exe /c run_all.bat
```

---

## 📦 Structure du démarrage automatique `run_all.bat`

- Ouvre une fenêtre pour l’API (`run_api.bat`)
- Ouvre une fenêtre pour Gradio (`run_gradio.bat`)
- Ouvre une fenêtre pour tester 15 images (`run_test.bat`)
- Chaque fenêtre reste ouverte indépendamment

---

## 📁 Dossiers importants

| Dossier                | Contenu                                              |
|------------------------|------------------------------------------------------|
| `test_images/`         | Images de test envoyées à l’API                      |
| `outputs/predictions/` | Masques prédits pour les images batch               |
| `outputs/logs/`        | Fichier `inference_log.csv` avec temps + metadata    |

---

## ✅ Bonnes pratiques

- Toujours commencer par `run_api.bat`
- Tester `http://127.0.0.1:8000/docs` dans un navigateur avant `run_test.bat`
- Ne jamais fermer la fenêtre API tant que Gradio ou tests sont en cours

---

## 🧠 À venir ?

- Ajout d’une légende de classes dans Gradio 🎨
- Visualisation live du log `inference_log.csv`
- Déploiement EC2 et Hugging Face

---

© Projet P08 — Ingénieur IA OpenClassrooms – 2025
