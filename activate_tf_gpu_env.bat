@echo off
echo 🔁 Activation de l'environnement tf_gpu_env...
call venv_p8_gpu\Scripts\activate.bat
python check_cuda_config.py
pause
