# api/inference.py

import numpy as np
import cv2
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
