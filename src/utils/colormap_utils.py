# src/utils/colormap_utils.py
import sys, os
import numpy as np
from PIL import Image



# PALETTE_P8 = {
#     0: (128, 64, 128),   # Flat
#     1: (70, 70, 70),     # Construction
#     2: (102, 102, 156),  # Object
#     3: (107, 142, 35),   # Nature
#     4: (70, 130, 180),   # Sky
#     5: (220, 20, 60),    # Human
#     6: (0, 0, 142),      # Vehicle
#     7: (0, 0, 0),        # Ignore
# }

# CLASS_NAMES_P8 = [
#     "Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle", "Ignore"
# ]

PALETTE_P8 = {
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
    "Flat", "Object", "Nature", "Construction", "Sky", "Vehicle", "Human", "Ignore"
]

def mask_to_colormap(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE_P8.items():
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask, mode="RGB")
