
# 🧹 Script clean_gpu_cache.sh — Libérer la mémoire GPU (NVIDIA)

Ce script permet de **libérer manuellement la mémoire GPU** utilisée par TensorFlow, Keras ou d'autres frameworks, **sans redémarrer la machine**.

## 📍 Quand l’utiliser ?

✅ Recommandé si :
- Tu fais plusieurs entraînements à la suite dans le même terminal
- Tu changes de modèle (ex: UNet → VGG16 → Optuna)
- Tu as une erreur de type `CUDA out of memory` alors qu’aucun entraînement n’est en cours

⚠️ Attention :
- Il termine tous les processus qui utilisent `/dev/nvidia*`
- À ne pas utiliser en prod avec d'autres utilisateurs ou scripts GPU critiques

## 🚀 Utilisation

Depuis le dossier `scripts/` :
```bash
bash clean_gpu_cache.sh
```

Tu verras :
- Les processus GPU en cours
- Leur arrêt forcé s’ils existent
- Un état de `nvidia-smi` avant/après

## 💡 Alternative douce dans Python (si tu préfères) :
```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

Ou :
```python
from numba import cuda
cuda.select_device(0)
cuda.close()
```

Mais ces solutions sont moins efficaces si plusieurs processus ou sessions sont en mémoire.

---

© Projet P8 - OpenClassrooms | Clean GPU 🧠
