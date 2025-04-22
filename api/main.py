# api/main.py

import io
import time
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from api.model_loader import load_segmentation_model
from api.inference import predict_mask, log_inference

app = FastAPI(title="API Segmentation P8")

# Middleware pour permettre à Gradio / autres clients d’appeler l’API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle global au démarrage
model = load_segmentation_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Route principale de prédiction.
    Accepte une image, renvoie le mask prédicté (Base64).
    """
    # Lire l’image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    start = time.time()
    mask = predict_mask(model, image)
    duration = time.time() - start

    # Convertir le mask en image couleur (palette segmentée)
    mask_img = Image.fromarray(mask.astype("uint8"), mode="L").resize(image.size)

    # Encode mask en base64 pour réponse JSON
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Log de l’inférence
    log_inference(file.filename, duration)

    return {
        "filename": file.filename,
        "inference_time": round(duration, 4),
        "mask_base64": mask_base64,
    }

# Pour tester en local : uvicorn api.main:app --reload

# Puis POST via Postman ou curl :
# curl -X POST -F "file=@mon_image.png" http://localhost:8000/predict
