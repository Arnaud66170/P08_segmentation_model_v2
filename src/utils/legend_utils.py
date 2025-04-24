import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

PALETTE_P8 = {
    0: (128, 64, 128),   # Flat
    1: (70, 70, 70),     # Construction
    2: (102, 102, 156),  # Object
    3: (107, 142, 35),   # Nature
    4: (70, 130, 180),   # Sky
    5: (220, 20, 60),    # Human
    6: (0, 0, 142),      # Vehicle
    7: (0, 0, 0),        # Ignore
}

CLASS_NAMES_P8 = [
    "Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle", "Ignore"
]



def generate_legend_image(palette: dict = PALETTE_P8, class_names: list = CLASS_NAMES_P8) -> Image.Image:
    """
    Génère une image PIL contenant la légende des classes à partir d'une palette RGB.
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(3, 0.6 * n), dpi=100)

    for i in range(n):
        color = tuple(np.array(palette[i]) / 255.0)
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.1, i + 0.5, class_names[i], va='center', fontsize=10)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, n)
    ax.axis('off')
    fig.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(img)
