# api/model_loader.py

import os
import sys

# Ajoute la racine du projet au PYTHONPATH pour que src/ soit importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras.models import load_model
from src.model_training.metrics import iou_score, dice_coef
# weighted_focal_loss pas nécessaire ici si on compile=False

# Dictionnaire des chemins vers les modèles sauvegardés
MODEL_PATHS = {
    "unet_mini": "notebooks/models/unet_mini_npz_256x256_bs8_ep40.h5",
    # "vgg16": "notebooks/models/unet_vgg16_256x256_bs4_ep40.h5",
    # "mobilenetv2": "notebooks/models/unet_mobilenetv2_256x256_bs4_ep40.h5",
}

# Objets personnalisés utilisés lors de l'entraînement (pas de loss ici)
CUSTOM_OBJECTS = {
    "iou_score": iou_score,
    "dice_coef": dice_coef
}

def load_segmentation_model(model_name: str = "unet_mini"):
    """
    Charge un modèle de segmentation pour prédiction.
    Ne compile pas le modèle car l'inférence ne nécessite pas la loss.

    Args:
        model_name (str): Nom du modèle à charger

    Returns:
        keras.Model: Modèle Keras prêt à l'emploi pour .predict()
    """
    if model_name not in MODEL_PATHS:
        print(f"[WARN] Modèle inconnu '{model_name}', fallback sur 'unet_mini'")
        model_name = "unet_mini"

    model_path = MODEL_PATHS[model_name]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print(f"[INFO] Chargement du modèle : {model_name} depuis {model_path}")
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)

    return model
