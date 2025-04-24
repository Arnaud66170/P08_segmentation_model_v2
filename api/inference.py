# api/inference.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import collections

from src.utils.colormap_utils import mask_to_colormap
from src.model_training.metrics import iou_score, dice_coef
from tensorflow.keras.models import load_model

TARGET_SIZE = (256, 256)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return np.expand_dims(image_array.astype(np.float32), axis=0)

def predict_mask(model, image: Image.Image) -> np.ndarray:
    x = preprocess_image(image)
    prediction = model.predict(x, verbose=0)[0]
    mask = np.argmax(prediction, axis=-1)

    print("[DEBUG] Classes brutes dans le mask :", collections.Counter(mask.flatten()))

    remap_indices = {
        7: 5,  # noir → Human (piéton)
        6: 6,  # bleu nuit → Vehicle
        4: 7,  # bleu clair → Ignore (capot)
        2: 0,  # bleu-gris → Flat (route)
        3: 2,  # vert olive → Object (panneau)
        1: 3,  # gris foncé → Nature (arbres)
        5: 4,  # rouge vif → Sky (ciel)
        0: 1,  # violet foncé → Construction (bâtiment)
    }

    # remap_indices = {
    #     3: 0,  # prédiction brute 3 → Flat
    #     5: 1,  # prédiction brute 5 → Object
    #     4: 2,  # prédiction brute 4 → Nature
    #     2: 3,  # prédiction brute 2 → Construction
    #     0: 4,  # prédiction brute 0 → Sky (corrigé, si ça correspond)
    #     6: 5,  # prédiction brute 6 → Vehicle
    #     1: 6,  # prédiction brute 1 → Human
    #     7: 7   # prédiction brute 7 → Ignore
    # }
    
    mask_remapped = np.zeros_like(mask)
    for src_class, tgt_class in remap_indices.items():
        mask_remapped[mask == src_class] = tgt_class

    return mask_remapped

def log_inference(filename: str, inference_time: float):
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

if __name__ == "__main__":
    model_path = "notebooks/models/unet_mini_npz_256x256_bs8_ep40.h5"
    model = load_model(model_path, custom_objects={"iou_score": iou_score, "dice_coef": dice_coef})

    image_path = "test_images/munich_000109_000019_leftImg8bit.png"
    image = Image.open(image_path)

    mask = predict_mask(model, image)
    print("[DEBUG] Top classes :", collections.Counter(mask.flatten()).most_common(10))

    color_mask = mask_to_colormap(mask)
    color_mask.show()
