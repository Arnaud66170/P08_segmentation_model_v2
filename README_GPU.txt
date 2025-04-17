📘 README_GPU.txt — Mise en place d'un environnement TensorFlow GPU (TF 2.10.1)

Ce fichier aide à (ré)activer un environnement compatible GPU.

────────────────────────────────────────────────────────────
1. Active l'environnement Python GPU
────────────────────────────────────────────────────────────
Ouvrir un terminal (CMD, PowerShell) à la racine du projet :

    > call activate_tf_gpu_env.bat

Ce script :
✔️ Active l'environnement virtuel `venv_p8_gpu`
✔️ Vérifie la présence du GPU via TensorFlow

────────────────────────────────────────────────────────────
2. Installer les dépendances GPU-compatibles
────────────────────────────────────────────────────────────
Si c'est la première activation ou après suppression de TensorFlow :

    > pip install -r requirements_gpu.txt

Cela installe :
- tensorflow==2.10.1 (compatible CUDA 11.2 + cuDNN 8.1)
- mlflow, optuna, albumentations, scikit-learn, etc.

────────────────────────────────────────────────────────────
3. Vérifier que le GPU est utilisé
────────────────────────────────────────────────────────────
On peut relancer le test :

    > python check_cuda_config.py

On doit voir :
- Un GPU listé
- CUDA activé
- Device : NVIDIA GTX 1060

────────────────────────────────────────────────────────────
4. Remarques
────────────────────────────────────────────────────────────
- Ne jamais faire cette manip en plein entraînement d'un modèle.
- Ne jamais mélanger TensorFlow CPU (2.15) et GPU (2.10.1).
- Toujours activer le bon kernel dans Jupyter : `tf_gpu_env`.

────────────────────────────────────────────────────────────
On est maintenant prêt à améliorer les temps d'entraînement !
