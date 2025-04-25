# Base image officielle pour Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers de dépendances en premier (optimisation du cache Docker)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu de ton projet
COPY . .

# Exposer le port de l’API FastAPI
EXPOSE 8000

# Commande de lancement de l’API avec uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
