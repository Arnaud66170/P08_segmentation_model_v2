# tests/test_api.py

import io
import base64
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import MagicMock

client = TestClient(app)

# ✅ Mocker proprement le modèle si en mode test
from api import main
main.model = MagicMock()
main.model.predict.return_value = None  # tu peux affiner si besoin

def test_predict_endpoint_returns_mask():
    # Créer une image RGB factice (256x256)
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Simuler un POST vers /predict
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", img_byte_arr, "image/png")},
    )

    assert response.status_code == 200
    json_data = response.json()

    assert "mask_base64" in json_data
    assert "inference_time" in json_data
    assert "filename" in json_data
    assert json_data["filename"] == "test_image.png"

    # Vérifier que le champ mask_base64 est bien décodable
    base64.b64decode(json_data["mask_base64"])
