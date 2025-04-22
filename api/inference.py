# api/inference.py

import numpy as np
import cv2
import os
from PIL import Image
from datetime import datetime
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model("models/best_model_unet.h5", compile=False)

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256)) / 255.0
    return np.expand_dims(img, axis=0)

def predict_mask(image_bytes):
    img = preprocess_image(image_bytes)
    pred = model.predict(img)[0]
    return (pred > 0.5).astype(np.uint8)

# api/inference.py

import os
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd

# Dimensions du modèle UNet Mini entraîné
TARGET_SIZE = (256, 256)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prépare une image pour la prédiction.
    Resize et normalise les valeurs.

    Args:
        image (PIL.Image): image d'entrée

    Returns:
        np.ndarray: image prête pour model.predict
    """
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0  # normalisation
    if image_array.ndim == 2:  # grayscale -> RGB
        image_array = np.stack([image_array]*3, axis=-1)
    image_array = image_array.astype(np.float32)
    return np.expand_dims(image_array, axis=0)  # shape (1, H, W, 3)


def predict_mask(model, image: Image.Image) -> np.ndarray:
    """
    Prédit le mask segmenté à partir d’une image.

    Args:
        model: modèle Keras
        image (PIL.Image): image d'entrée

    Returns:
        np.ndarray: mask prédicté (2D)
    """
    x = preprocess_image(image)
    prediction = model.predict(x, verbose=0)[0]  # (H, W, C)
    mask = np.argmax(prediction, axis=-1)  # (H, W)
    return mask


def log_inference(filename: str, inference_time: float):
    """
    Enregistre les détails de l'inférence dans un fichier CSV.

    Args:
        filename (str): nom du fichier/image
        inference_time (float): temps en secondes
    """
    os.makedirs("outputs/logs", exist_ok=True)
    log_path = "outputs/logs/inference_log.csv"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "inference_time_s": round(inference_time, 4)
    }

    df = pd.DataFrame([log_data])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", index=False, header=False)
    else:
        df.to_csv(log_path, mode="w", index=False, header=True)
