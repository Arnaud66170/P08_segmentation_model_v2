# src/model_training/optimize_unet_with_optuna.py

import time
import optuna
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import mlflow
import mlflow.keras

from model_training.train_unet_backbones import build_unet_backbone
from model_training.metrics import iou_score, dice_coef
from utils.logger import log_step
from utils.utils import plot_history

ARTIFACTS_DIR = Path("models")
ARTIFACTS_DIR.mkdir(exist_ok=True)

@log_step
def run_optuna_for_backbone(backbone_name, X_train, Y_train, X_val, Y_val, n_trials=20, timeout=None):

    def objective(trial):
        params = {
            'dropout': trial.suggest_float("dropout", 0.1, 0.5),
            'learning_rate': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical("batch_size", [4, 8, 16]),
            'epochs': trial.suggest_int("epochs", 20, 50),
        }

        model_name = f"optuna_{backbone_name}_bs{params['batch_size']}_ep{params['epochs']}_lr{params['learning_rate']:.0e}"
        model_path = ARTIFACTS_DIR / f"{model_name}.h5"
        history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
        plot_path = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
        csv_path = ARTIFACTS_DIR / f"{model_name}_history.csv"

        model = build_unet_backbone(backbone_name, input_shape=(256, 256, 3), num_classes=8)

        # Ajout dropout si requis dans le decoder
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                model = keras.Sequential([model, layers.Dropout(params['dropout'])])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", iou_score, dice_coef]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True),
            keras.callbacks.CSVLogger(str(csv_path))
        ]

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                **params,
                "backbone": backbone_name,
                "optuna": True
            })

            start = time.time()
            history_obj = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=callbacks,
                shuffle=True,
                verbose=0
            )
            end = time.time()

            duration = round(end - start, 2)
            mlflow.log_metric("train_time_sec", duration)

            history = history_obj.history
            joblib.dump(history, history_path)
            pd.DataFrame(history).to_csv(csv_path, index=False)
            plot_history(history_obj, plot_path)

            mlflow.keras.log_model(model, model_name)
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(csv_path))

            final_score = max(history["val_iou_score"])
            mlflow.log_metric("best_val_iou", final_score)

        return final_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("\n🏆 Meilleur score IoU :", study.best_value)
    print("📋 Meilleurs paramètres :", study.best_params)

    results_path = ARTIFACTS_DIR / f"optuna_results_{backbone_name}.csv"
trial_df = study.trials_dataframe()
trial_df.to_csv(results_path, index=False)
print(f"
Résultats des essais sauvegardés : {results_path}")

            results_path = ARTIFACTS_DIR / f"optuna_results_{backbone_name}.csv"
        trial_df = study.trials_dataframe()
        trial_df.to_csv(results_path, index=False)
        print(f"
Résultats des essais sauvegardés : {results_path}")

        return study