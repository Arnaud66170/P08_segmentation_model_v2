# src/model_training/train_unet.py

import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import mlflow
import mlflow.keras

from model_training.metrics import iou_score, dice_coef
from utils.mlflow_manager import mlflow_logging_decorator
from utils.logger import log_step
from utils.utils import plot_history

ARTIFACTS_DIR = Path("models")
ARTIFACTS_DIR.mkdir(exist_ok=True)

@mlflow_logging_decorator
@log_step
def unet_mini(input_shape=(256, 256, 3), num_classes=8):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c4)
    return keras.Model(inputs, outputs)

@mlflow_logging_decorator
@log_step
def train_unet_model_from_npz(X_train, Y_train, X_val, Y_val,
                              force_retrain=False,
                              img_size=(256, 256),
                              epochs=20,
                              batch_size=8,
                              use_early_stopping=True,
                              num_classes=8):

    model_name = f"unet_mini_npz_{img_size[0]}x{img_size[1]}_bs{batch_size}_ep{epochs}"
    model_path   = ARTIFACTS_DIR / f"{model_name}.h5"
    history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
    plot_path    = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
    csv_path     = ARTIFACTS_DIR / f"{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        print(f"[INFO] Modèle existant détecté : {model_path}")
        model = keras.models.load_model(model_path, custom_objects={"iou_score": iou_score, "dice_coef": dice_coef})
        history = joblib.load(history_path)
        return model, history

    print("[INFO] Initialisation du modèle UNet Mini...")
    model = unet_mini(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', iou_score, dice_coef]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True)
    ]
    if use_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
    callbacks.append(keras.callbacks.CSVLogger(str(csv_path)))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "input_shape": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "force_retrain": force_retrain,
            "source": ".npz"
        })

        start = time.time()
        history_obj = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        end = time.time()
        duration = round(end - start, 2)
        mlflow.log_metric("train_time_sec", duration)

        joblib.dump(history_obj.history, history_path)
        pd.DataFrame(history_obj.history).to_csv(csv_path, index=False)
        plot_history(history_obj, plot_path)

        mlflow.keras.log_model(model, model_name)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(csv_path))

        for epoch in range(len(history_obj.history['loss'])):
            mlflow.log_metric("val_accuracy", history_obj.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history_obj.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_iou_score", history_obj.history['val_iou_score'][epoch], step=epoch)
            mlflow.log_metric("val_dice_coef", history_obj.history['val_dice_coef'][epoch], step=epoch)

    return model, history_obj.history