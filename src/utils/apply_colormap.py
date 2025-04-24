# src/utils/apply_colormap.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 👈 Celle-là est obligatoire

def apply_colormap(mask_array, colormap="nipy_spectral", vmin=0, vmax=7):
    '''
    Applique une colormap matplotlib à un masque de segmentation (2D numpy array).

    Args:
        mask_array (np.ndarray): masque (H, W) avec valeurs discrètes (0 à N)
        colormap (str): nom de la colormap matplotlib
        vmin (int): valeur min pour normalisation
        vmax (int): valeur max pour normalisation

    Returns:
        PIL.Image.Image : image colorisée
    '''
    cmap = plt.get_cmap(colormap)
    normed = np.clip((mask_array - vmin) / (vmax - vmin), 0, 1)
    color_array = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(color_array)