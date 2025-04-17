# src/utils/guardrail.py

from pathlib import Path
import importlib
import tensorflow as tf
from tensorflow import keras

def check_paths_exist(paths):
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"❌ Chemin introuvable : {path}")
        else:
            print(f"✅ Chemin OK : {path}")

def check_imports(module_list):
    for m in module_list:
        try:
            importlib.import_module(m)
            print(f"✅ Import OK : {m}")
        except Exception as e:
            raise ImportError(f"❌ Échec import '{m}' : {e}")

def check_models_validity(model_dir, custom_objects=None):
    model_dir = Path(model_dir)
    models = list(model_dir.glob("*.h5"))
    if not models:
        print("⚠️ Aucun fichier modèle .h5 trouvé dans :", model_dir)
    
    for m in models:
        try:
            keras.models.load_model(m, custom_objects=custom_objects or {})
            print(f"✅ Modèle chargeable : {m.name}")
        except Exception as e:
            print(f"❌ Modèle corrompu ou incompatible : {m.name} → {e}")
