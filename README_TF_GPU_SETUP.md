
# ⚙️ Activation du Mode GPU pour TensorFlow (GTX 1060, CUDA 11.2, cuDNN 8.1)

Ce document décrit comment activer **l’accélération GPU** pour ton projet TensorFlow sur une machine équipée d’une **GTX 1060** sous Windows ou Linux.

---

## 📦 1. Installer la bonne version de TensorFlow

La dernière version compatible avec CUDA 11.2 et cuDNN 8.1 est :
```bash
pip install tensorflow==2.10.1
```

> ❗ TensorFlow 2.11+ ne supporte plus le GPU sous Windows (utilise TF 2.10.1 ou inférieur)

---

## 💾 2. Installer les bibliothèques NVIDIA nécessaires

### Sous Windows :
1. Installe **CUDA Toolkit 11.2** : https://developer.nvidia.com/cuda-11.2.0-download-archive
2. Installe **cuDNN 8.1 pour CUDA 11.2** :
   - Lien : https://developer.nvidia.com/rdp/cudnn-archive
   - Décompresse le contenu dans le dossier `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

### Sous Linux (Ubuntu) :
```bash
sudo apt install nvidia-cuda-toolkit
```
> Puis vérifie avec `nvidia-smi` et assure-toi que la version CUDA = 11.2

---

## 🧪 3. Vérifier que TensorFlow détecte bien le GPU

Dans un notebook ou script Python :
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
Ou plus simplement :
```python
import tensorflow as tf
print("GPU disponible :", tf.config.list_physical_devices('GPU'))
```

Tu dois voir ta carte GTX 1060 apparaître.

---

## 🚀 4. Exécuter ton notebook avec GPU

Si tu utilises Jupyter ou VSCode, assure-toi que ton kernel Python est bien celui activé avec TensorFlow GPU.

Tu peux le forcer avec un batch d’environnement :
```bash
# Exemple Windows (à adapter)
call scripts\launch_tf_gpu_env.bat
```
Et un script pour enregistrer le kernel :
```bash
python scripts\register_kernel.py
```

---

## 📌 5. (Optionnel) Activer le mode optimisé

Ajoute ce code au début de ton script pour accélérer les calculs :

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Active XLA et mixed precision
tf.config.optimizer.set_jit(True)  # XLA
mixed_precision.set_global_policy('mixed_float16')
```

---

## 🧠 Notes

- GTX 1060 = architecture Pascal = supporte très bien CUDA 11.2
- Ne mélange pas plusieurs versions de CUDA/cuDNN
- Si TensorFlow ne voit pas le GPU → vérifier `nvidia-smi`, PATH, et `DLL` cuDNN sous Windows

---

© Projet P8 - OpenClassrooms | MLOps GPU Ready ⚡
