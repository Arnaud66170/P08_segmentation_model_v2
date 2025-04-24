# debug_tf_load.py

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.utils.legend_utils import generate_legend_image
from collections import Counter
# from src.utils.class_definitions import CLASS_NAMES_P8

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.utils.legend_utils import generate_legend_image
from collections import Counter

# === Palette officielle P8 ===
palette = {
    0: (128, 64, 128),   # Flat
    1: (102, 102, 156),  # Object
    2: (107, 142, 35),   # Nature
    3: (70, 70, 70),     # Construction
    4: (70, 130, 180),   # Sky
    5: (0, 0, 142),      # Vehicle
    6: (220, 20, 60),    # Human
    7: (0, 0, 0),        # Ignore
}

CLASS_NAMES_P8 = [
    "Flat", "Object", "Nature", "Construction",
    "Sky", "Vehicle", "Human", "Ignore"
]

legend_img = generate_legend_image()
legend_img.show()

def mask_to_colormap(mask: np.ndarray) -> Image.Image:
    """Transforme un masque de classes (0-7) en image RGB selon la palette."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask, mode="RGB")

# === Paramètres ===
MODEL_PATH = "notebooks/models/unet_mini_npz_256x256_bs8_ep40.h5"
IMAGE_PATH = "test_images/calibration_p8_mire.png"
INPUT_SIZE = (256, 256)

print("[INFO] Chargement du modèle...")
model = load_model(MODEL_PATH)

print("[INFO] Chargement de l’image...")
img = Image.open(IMAGE_PATH).convert("RGB")
img_resized = img.resize(INPUT_SIZE)
x = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

print("[INFO] Prédiction du masque...")
pred = model.predict(x)[0]
mask = np.argmax(pred, axis=-1)  # (H, W)

print("[🔍 Analyse automatique des blocs de mire]")
block_size = 64
remap_indices = {}

for i in range(8):
    row = i // 4
    col = i % 4
    x_start = col * block_size
    y_start = row * block_size
    patch = mask[y_start:y_start + block_size, x_start:x_start + block_size]
    values, counts = np.unique(patch, return_counts=True)
    major_class = int(values[np.argmax(counts)])
    print(f"Case {i} → prédiction brute : {major_class}")
    remap_indices[major_class] = i

# 🧪 Debug : distribution des classes AVANT remapping
print("[DEBUG] Raw mask (avant remap) :")
print(np.unique(mask, return_counts=True))
print("[DEBUG] Top classes dans le mask :")
print(Counter(mask.flatten()).most_common())

🔁 Remappage
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
for k, v in remap_indices.items():
    mask_remapped[mask == k] = v
mask = mask_remapped

print("\n[🔊] Vérification couverture des classes P8")
for class_id, count in sorted(Counter(mask.flatten()).items()):
    if class_id < len(CLASS_NAMES_P8):
        print(f"Classe {class_id:2d} : {CLASS_NAMES_P8[class_id]:<14} — {count} px")
    else:
        print(f"Classe {class_id:2d} : [Inconnue] — {count} px")

# === Affichage : image originale + mask brut + mask colorisé ===
mask_rgb = mask_to_colormap(mask)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_resized)
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=7)
axes[1].set_title("Masque brut (0-7)")
axes[1].axis("off")

axes[2].imshow(mask_rgb)
axes[2].set_title("Masque colorisé (palette P8)")
axes[2].axis("off")

plt.tight_layout()
plt.show()






# # === Palette P8 : 8 classes avec leurs couleurs spécifiques ===
# palette = {
#     0: (128, 64, 128),   # flat
#     1: (70, 70, 70),     # construction
#     2: (102, 102, 156),  # object
#     3: (107, 142, 35),   # nature
#     4: (70, 130, 180),   # sky
#     5: (220, 20, 60),    # human
#     6: (0, 0, 142),      # vehicle
#     7: (0, 0, 0),        # ignore
# }

# # class_names = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle", "Ignore"]

# CLASS_NAMES_P8 = [
#     "Flat",         # 0
#     "Object",       # 1
#     "Nature",       # 2
#     "Construction", # 3
#     "Sky",          # 4
#     "Vehicle",      # 5
#     "Human",        # 6
#     "Ignore"        # 7
# ]

# legend_img = generate_legend_image()
# legend_img.show()

# def mask_to_colormap(mask: np.ndarray) -> Image.Image:
#     """Transforme un masque de classes (0-7) en image RGB selon la palette."""
#     h, w = mask.shape
#     color_mask = np.zeros((h, w, 3), dtype=np.uint8)
#     for class_id, color in palette.items():
#         color_mask[mask == class_id] = color
#     return Image.fromarray(color_mask, mode="RGB")

# # === Paramètres ===
# MODEL_PATH = "notebooks/models/unet_mini_npz_256x256_bs8_ep40.h5"
# # IMAGE_PATH = "test_images/munich_000109_000019_leftImg8bit.png"
# IMAGE_PATH = "test_images/calibration_p8_mire.png"

# INPUT_SIZE = (256, 256)

# # === Chargement du modèle et de l'image ===
# print("[INFO] Chargement du modèle...")
# model = load_model(MODEL_PATH)

# print("[INFO] Chargement de l’image...")
# img = Image.open(IMAGE_PATH).convert("RGB")
# img_resized = img.resize(INPUT_SIZE)
# x = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

# # === Prédiction du masque ===
# print("[INFO] Prédiction du masque...")
# pred = model.predict(x)[0]
# mask = np.argmax(pred, axis=-1)  # (H, W)

# # Dimensions des blocs (ex: 64x64 pour une image 256x256)
# block_size = 64
# bloc_preds = {}

# print("[🔍 Analyse de la mire par bloc]")
# for i in range(2):  # lignes
#     for j in range(4):  # colonnes
#         x1, x2 = j * block_size, (j + 1) * block_size
#         y1, y2 = i * block_size, (i + 1) * block_size
#         bloc = mask[y1:y2, x1:x2]
#         most_common = np.bincount(bloc.flatten()).argmax()
#         bloc_index = i * 4 + j
#         bloc_preds[most_common] = bloc_index
#         print(f"Case {bloc_index} → prédiction brute : {most_common}")

# # 🧪 Debug : distribution des classes AVANT remapping
# print("[DEBUG] Raw mask (avant remap) :")
# print(np.unique(mask, return_counts=True))

# # 🔍 Analyse rapide des classes prédominantes
# import collections
# print("[DEBUG] Top classes dans le mask :")
# print(collections.Counter(mask.flatten()).most_common())

# remap_indices = {
#     5: 0,  # prédiction brute 5 → Flat
#     1: 1,  # prédiction brute 1 → Object
#     2: 2,  # prédiction brute 2 → Nature
#     3: 3,  # prédiction brute 3 → Construction
#     4: 4,  # prédiction brute 4 → Sky
#     6: 5,  # prédiction brute 6 → Vehicle
#     0: 6,  # prédiction brute 0 → Human
#     7: 7,  # prédiction brute 7 → Ignore
# }


# # 🔁 Remappage
# # remap_indices = {
# #     0: 6, 1: 5, 2: 4, 3: 3,
# #     4: 2, 5: 1, 6: 0, 7: 7,
# # }
# # remap_indices = {
# #     0: 5,  # prédiction 0 = human (rouge vif)
# #     1: 1,  # prédiction 1 = construction
# #     2: 0,  # prédiction 2 = flat
# #     3: 2,  # prédiction 3 = object
# #     4: 4,  # prédiction 4 = sky
# #     5: 6,  # prédiction 5 = vehicle
# #     6: 7,  # prédiction 6 = ignore (capot)
# #     7: 3,  # prédiction 7 = nature (dominant → herbe/arbres ?)
# # }

# mask_remapped = np.zeros_like(mask)
# for k, v in remap_indices.items():
#     mask_remapped[mask == k] = v

# mask = mask_remapped

# print("\n[🧪 Vérification couverture des classes P8]")
# for class_id, count in sorted(Counter(mask.flatten()).items()):
#     if class_id < len(CLASS_NAMES_P8):
#         print(f"Classe {class_id:2d} : {CLASS_NAMES_P8[class_id]:<14} — {count} px")
#     else:
#         print(f"Classe {class_id:2d} : [Inconnue] — {count} px")


# # === Affichage : image originale + mask brut + mask colorisé ===
# mask_rgb = mask_to_colormap(mask)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(img_resized)
# axes[0].set_title("Image originale")
# axes[0].axis("off")

# axes[1].imshow(mask, cmap="gray", vmin=0, vmax=7)
# axes[1].set_title("Masque brut (0-7)")
# axes[1].axis("off")

# axes[2].imshow(mask_rgb)
# axes[2].set_title("Masque colorisé (palette P8)")
# axes[2].axis("off")

# plt.tight_layout()
# plt.show()

# # lancement du debug dans terminal cmd.exe
# # cd C:\Users\motar\Desktop\1-openclassrooms\AI_Engineer\1-projets\P08\P08_segmentation
# # python debug_tf_load.py
