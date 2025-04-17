import tensorflow as tf
from tensorflow.python.client import device_lib

print("🔍 TensorFlow version :", tf.__version__)
print("🧠 Compilation CUDA activée :", tf.test.is_built_with_cuda())
print("📦 GPU visible :", tf.config.list_physical_devices('GPU'))
print("\n🖥️ Détails complets des devices :")
print(device_lib.list_local_devices())
