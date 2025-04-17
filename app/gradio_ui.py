# app/graduio_ui.py

import gradio as gr
import requests
from PIL import Image
import numpy as np
import io

API_URL = "http://<IP_EC2>:8000/predict"

def call_api(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    response = requests.post(API_URL, files={"file": buffered.getvalue()})
    mask = np.array(response.json()["mask"]) * 255
    return image, Image.fromarray(mask.astype(np.uint8))

demo = gr.Interface(
    fn=call_api,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Original"), gr.Image(label="Mask prédite")],
    title="Segmentation Demo"
)

if __name__ == "__main__":
    demo.launch()
