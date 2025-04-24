# scripts/test_predict_local.py

import requests
from PIL import Image
import io
import base64
import os

API_URL = "http://127.0.0.1:8000/predict"
TEST_FOLDER = "test_images"
OUTPUT_FOLDER = "outputs/predictions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("[INFO] Test local de prédiction d'images lancé...")
print(f"[INFO] Dossier testé : {TEST_FOLDER}")
print(f"[INFO] 🔗 URL API : {API_URL}")
print("-" * 60)

for filename in sorted(os.listdir(TEST_FOLDER)):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(TEST_FOLDER, filename)

    try:
        with open(img_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            print("[DEBUG] Réponse API :", data)
            mask_data = base64.b64decode(data["mask_base64"])
            mask_img = Image.open(io.BytesIO(mask_data))

            out_path = os.path.join(OUTPUT_FOLDER, f"pred_{filename}")
            mask_img.save(out_path)
            print(f"[✓] {filename} traité → {out_path} ({data['inference_time']} sec)")
        else:
            print(f"[✗] Erreur HTTP {response.status_code} sur {filename} : {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"[] ERREUR : l'API FastAPI n'est pas disponible à {API_URL}")
        print("[] Assure-toi que le serveur FastAPI est lancé avec : uvicorn api.main:app --reload")
        break
