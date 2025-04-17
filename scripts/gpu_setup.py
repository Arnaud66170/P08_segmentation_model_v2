
import tensorflow as tf
from tensorflow.keras import mixed_precision

def enable_gpu_boost():
    """
    Active le mode optimisé GPU : XLA + Mixed Precision + résumé GPU
    """
    print("✅ TensorFlow version :", tf.__version__)
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🟢 GPU détecté : {gpus[0].name}")
    else:
        print("🔴 Aucun GPU détecté, entraînement en mode CPU.")
        return

    # ⚡ XLA JIT compilation
    # tf.config.optimizer.set_jit(True)

    # 🎯 Mixed Precision float16
    # mixed_precision.set_global_policy("mixed_float16")
    mixed_precision.set_global_policy("float32")


    # 📊 Affichage mémoire GPU
    try:
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        for d in devices:
            if d.device_type == 'GPU':
                mem_gb = round(d.memory_limit / 1024**3, 2)
                print(f"🔧 GPU utilisé : {d.name} | Mémoire : {mem_gb} GB")
    except Exception as e:
        print(f"⚠️ Impossible d'afficher les infos GPU : {e}")
