# app/gradio_ui.py

import requests
import gradio as gr
import base64
from PIL import Image, ImageEnhance
import io
import os
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

from src.utils.legend_utils import generate_legend_image

API_URL = "http://localhost:8000/predict"
history = deque(maxlen=5)

# ✅ Générer la légende depuis la palette P8
legend_img = generate_legend_image()

# Superposition d'un masque colorisé sur une image d’entrée avec un facteur de transparence alpha.
def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    image = image.convert("RGBA").resize(mask.size)
    mask = mask.convert("RGBA")
    blended = Image.blend(image, mask, alpha=alpha)
    return blended

def segment_image(image: Image.Image):
    try:
        # Conversion image en binaire pour envoi HTTP
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        files = {"file": ("input.png", buffered, "image/png")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        data = response.json()

        # Décodage du masque colorisé retourné (base64 → PIL.Image)
        mask_bytes = base64.b64decode(data["mask_base64"])
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")

        # ✅ Superposition
        overlay_img = overlay_mask_on_image(image, mask_image)

        return image, mask_image, overlay_img, f"{data['inference_time']} sec"

    except Exception as e:
        print(f"[ERREUR] {e}")
        return None, None, None, f"Erreur : {str(e)}"

def run_batch_test():
    folder = "test_images"
    results = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".png"):
            with open(os.path.join(folder, filename), "rb") as f:
                files = {"file": (filename, f, "image/png")}
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                time_taken = data["inference_time"]
                results.append((filename, f"{time_taken} sec"))
            else:
                results.append((filename, f"Erreur {response.status_code}"))

    # Export CSV
    os.makedirs("outputs/predictions", exist_ok=True)
    pd.DataFrame(results, columns=["filename", "inference_time"]).to_csv("outputs/predictions/batch_results.csv", index=False)
    return gr.update(value=results)

def read_inference_log():
    try:
        return pd.read_csv("outputs/logs/inference_log.csv")
    except:
        return pd.DataFrame(columns=["timestamp", "filename", "inference_time"])

def plot_inference_histogram():
    df = read_inference_log()
    if df.empty:
        return plt.figure()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["filename"], df["inference_time"], color="skyblue")
    ax.set_ylabel("Temps (s)")
    ax.set_xlabel("Image")
    ax.set_title("Temps d'inférence par image")
    ax.tick_params(axis='x', rotation=45)
    return fig

with gr.Blocks(title="Segmentation d'Images Urbaines") as demo:
    gr.Markdown("# 🧠 Segmentation d'Images Urbaines")
    gr.Markdown("Upload une image, et observe la magie de UNet Mini en action 🚀")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Image d'entrée")
        btn = gr.Button("Segmenter")

    with gr.Row():
        img_original = gr.Image(label="Image originale")
        img_mask = gr.Image(label="Mask prédit")
        img_overlay = gr.Image(label="Superposition (image + mask)")  # ✅ nouvelle sortie
        inf_time = gr.Textbox(label="Temps d'inférence")

    with gr.Accordion("🕓 Historique des 5 dernières prédictions", open=False):
        gallery = gr.Gallery(label="Historique", show_label=False, columns=5, rows=1)

    btn.click(fn=segment_image, inputs=input_image, outputs=[img_original, img_mask, img_overlay, inf_time])

    with gr.Row():
        btn_batch = gr.Button("Lancer le Batch Test (15 images)")
        batch_output = gr.Dataframe(headers=["Nom de l'image", "Temps d'inférence"],
                                    datatype=["str", "str"],
                                    label="Résultats batch",
                                    interactive=False)
        btn_batch.click(fn=run_batch_test, inputs=[], outputs=batch_output)

    with gr.Row():
        if os.path.exists("outputs/predictions/batch_results.csv"):
            gr.File(label="📥 Télécharger les résultats CSV", value="outputs/predictions/batch_results.csv")

    with gr.Accordion("🎨 Légende des classes segmentées", open=False):
        gr.Image(value=legend_img, label="Légende des couleurs", interactive=False)

    with gr.Accordion("📈 Journal d'inférence (inference_log.csv)", open=False):
        log_plot = gr.Plot(label="Histogramme des temps d'inférence")
        log_table = gr.Dataframe(label="Log des inférences")
        btn_reload_log = gr.Button("🔄 Rafraîchir le log")
        btn_reload_log.click(fn=lambda: (plot_inference_histogram(), read_inference_log()),
                             inputs=[], outputs=[log_plot, log_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
