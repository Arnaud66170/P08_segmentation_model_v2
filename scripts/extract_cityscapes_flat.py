# scripts/extract_cityscapes_flat.py

# - Ce script te permet :
#   - d’avoir un dataset plat et propre dans data/raw/
#   - de relancer immédiatement prepare_dataset() derrière
#   - d’assurer que chaque image a son mask (base commune)

import os
import shutil
from glob import glob
from tqdm import tqdm
import sys
from glob import glob
from tqdm import tqdm
import argparse

def extract_cityscapes_to_raw(root_img_dir, root_mask_dir, output_image_dir, output_mask_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 1. Récupération des chemins d'images et masks
    image_paths = glob(os.path.join(root_img_dir, "**", "*_leftImg8bit.png"), recursive=True)
    mask_paths = glob(os.path.join(root_mask_dir, "**", "*_gtFine_labelIds.png"), recursive=True)

    print(f"[INFO] {len(image_paths)} images trouvées")
    print(f"[INFO] {len(mask_paths)} masks trouvés")

    # 2. Dictionnaire pour faire correspondre images ↔ masks
    mask_dict = {os.path.basename(f).replace("_gtFine_labelIds.png", ""): f for f in mask_paths}

    nb_copies = 0
    for image_path in tqdm(image_paths, desc="Copie des images + masks"):
        base_name = os.path.basename(image_path).replace("_leftImg8bit.png", "")
        if base_name in mask_dict:
            new_img_name = f"{base_name}.png"
            new_mask_name = f"{base_name}.png"

            shutil.copy(image_path, os.path.join(output_image_dir, new_img_name))
            shutil.copy(mask_dict[base_name], os.path.join(output_mask_dir, new_mask_name))
            nb_copies += 1
        else:
            print(f"[WARNING] Aucun mask correspondant pour : {image_path}")

    print(f"[INFO] Total de paires copiées : {nb_copies}")

# === Bloc principal avec gestion --force ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction Cityscapes vers data/raw/ (images + masks)")
    parser.add_argument("img_src", help="Dossier source des images (leftImg8bit)")
    parser.add_argument("mask_src", help="Dossier source des masks (gtFine)")
    parser.add_argument("output_img_dir", help="Dossier de destination des images (ex: data/raw/images)")
    parser.add_argument("output_mask_dir", help="Dossier de destination des masks (ex: data/raw/masks)")
    parser.add_argument("--force", action="store_true", help="Forcer la réextraction même si des fichiers existent déjà")

    args = parser.parse_args()

    if not args.force:
        if os.path.exists(args.output_img_dir) and os.path.exists(args.output_mask_dir):
            has_imgs = len(os.listdir(args.output_img_dir)) > 0
            has_masks = len(os.listdir(args.output_mask_dir)) > 0

            if has_imgs and has_masks:
                print("[INFO] Les dossiers cibles existent déjà et contiennent des fichiers. Extraction annulée.")
                sys.exit(0)

    extract_cityscapes_to_raw(
        root_img_dir=args.img_src,
        root_mask_dir=args.mask_src,
        output_image_dir=args.output_img_dir,
        output_mask_dir=args.output_mask_dir
    )