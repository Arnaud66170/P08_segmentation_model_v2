# api/main.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import base64
import traceback
import numpy as np

from api.model_loader import load_segmentation_model
from api.inference import predict_mask, log_inference
from src.utils.colormap_utils import mask_to_colormap

def create_app():
    app = FastAPI(title="API Segmentation P8")

    # Middleware pour autoriser CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Chargement conditionnel du modèle global (UNet Mini par défaut)
    global model
    model = None
    if os.environ.get("DISABLE_MODEL_LOADING", "0") != "1":
        model = load_segmentation_model()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            if model is None:
                return {"error": "Modèle non chargé."}

            # Lecture de l'image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Prédiction
            start = time.time()
            mask = predict_mask(model, image)
            duration = time.time() - start

            # Debug console
            print("[DEBUG] 📏 Mask shape :", mask.shape)
            print("[DEBUG] 🎯 Classes uniques :", np.unique(mask))

            # Application de la colormap
            try:
                mask_img = mask_to_colormap(mask).resize(image.size)
            except Exception as colormap_error:
                print("🛑 ERREUR DANS mask_to_colormap")
                traceback.print_exc()
                return {"error": f"Erreur de colorisation : {str(colormap_error)}"}

            # Encodage Base64
            buffer = io.BytesIO()
            mask_img.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Logging
            log_inference(file.filename, duration)

            return {
                "filename": file.filename,
                "inference_time": round(duration, 4),
                "mask_base64": mask_base64
            }

        except Exception as e:
            print("🛑 ERREUR PENDANT /predict")
            traceback.print_exc()
            return {"error": f"Erreur serveur : {str(e)}"}

    return app

app = create_app()
