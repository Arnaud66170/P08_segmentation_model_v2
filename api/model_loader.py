# api/model_loader.py

import os
from tensorflow.keras.models import load_model
from src.utils.losses import weighted_focal_loss
from src.utils.metrics import iou_score, dice_coef

# Dictionnaire des chemins vers les modèles
MODEL_PATHS = {
    "unet_mini": "models/unet_mini_npz_256x256_bs8_ep40.h5",
    # "vgg16": "models/unet_vgg16_256x256_bs4_ep40.h5",
    # "mobilenetv2": "models/unet_mobilenetv2_256x256_bs4_ep40.h5",
}

# Dictionnaire des objets custom à injecter (pour charger les modèles compilés)
CUSTOM_OBJECTS = {
    "iou_score": iou_score,
    "dice_coef": dice_coef,
    "weighted_focal_loss": weighted_focal_loss,
}

def load_segmentation_model(model_name: str = "unet_mini"):
    """
    Charge un modèle de segmentation à partir du nom fourni.
    Par défaut : 'unet_mini'.

    Args:
        model_name (str): le nom du modèle à charger

    Returns:
        keras.Model: le modèle chargé
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
