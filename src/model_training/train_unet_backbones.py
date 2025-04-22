# src/model_training/train_unet_backbones.py

import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import mlflow
from tensorflow.keras.applications import VGG16, MobileNetV2, EfficientNetB0

from model_training.metrics import iou_score, dice_coef
from utils.mlflow_manager import mlflow_logging_decorator
from utils.logger import log_step
from utils.utils import plot_history

ARTIFACTS_DIR = Path("models")
ARTIFACTS_DIR.mkdir(exist_ok=True)

BACKBONES = {
    "vgg16": VGG16,
    "mobilenetv2": MobileNetV2,
    "efficientnetb0": EfficientNetB0
}

def safe_float(val):
    if isinstance(val, (tf.Tensor, np.generic)):
        return float(val)
    elif isinstance(val, (list, np.ndarray)):
        return [float(v) for v in val]
    return val

@mlflow_logging_decorator
@log_step
def build_unet_backbone(backbone_name, input_shape=(256, 256, 3), num_classes=8):
    base_model = BACKBONES[backbone_name](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    for layer in base_model.layers:
        layer.trainable = False

    inputs = base_model.input
    x = base_model.output

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return keras.Model(inputs, outputs)

@mlflow_logging_decorator
@log_step
def train_unet_with_backbone(backbone_name,
                             X_train, Y_train, X_val, Y_val,
                             force_retrain=False,
                             img_size=(256, 256),
                             epochs=40,
                             batch_size=8,
                             use_early_stopping=True,
                             num_classes=8):

    model_name = f"unet_{backbone_name}_{img_size[0]}x{img_size[1]}_bs{batch_size}_ep{epochs}"
    model_path   = ARTIFACTS_DIR / f"{model_name}.h5"
    history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
    plot_path    = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
    csv_path     = ARTIFACTS_DIR / f"{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        try:
            print(f"[INFO] Modèle existant pour {backbone_name} : reconstruction + chargement des poids...")
            model = build_unet_backbone(backbone_name, input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
            model.load_weights(model_path)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy", iou_score, dice_coef]
            )
            history = joblib.load(history_path)
            return model, history

        except Exception as e:
            print(f"[⚠️] Échec lors du chargement du modèle existant : {e}")
            print(f"[INFO] Suppression du fichier corrompu et réentraînement forcé.")
            model_path.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)

    print(f"[INFO] Initialisation du modèle UNet avec backbone : {backbone_name}")
    model = build_unet_backbone(backbone_name, input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", iou_score, dice_coef]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True, save_weights_only=True)
    ]
    if use_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
    callbacks.append(keras.callbacks.CSVLogger(str(csv_path)))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "backbone": backbone_name,
            "input_shape": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "force_retrain": force_retrain
        })
        mlflow.log_param("model_name", model_name)

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

        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(csv_path))

        for epoch in range(len(history_obj.history['loss'])):
            mlflow.log_metric("val_accuracy", safe_float(history_obj.history['val_accuracy'][epoch]), step=epoch)
            mlflow.log_metric("val_loss",     safe_float(history_obj.history['val_loss'][epoch]),     step=epoch)
            mlflow.log_metric("val_iou_score", safe_float(history_obj.history['val_iou_score'][epoch]), step=epoch)
            mlflow.log_metric("val_dice_coef", safe_float(history_obj.history['val_dice_coef'][epoch]), step=epoch)

    return model, history_obj.history
