# src/utils/losses.py

import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_focal_loss(class_weights, gamma=2.0):
    """
    Focal loss pondérée pour la segmentation multiclasses.

    Args:
        class_weights (list or array): poids associés à chaque classe
        gamma (float): facteur de focalisation (plus grand = plus focalisé)

    Returns:
        fonction de perte personnalisée
    """
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        # One-hot encoding sur la dernière dimension
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())

        cross_entropy = -y_true_oh * tf.math.log(y_pred)
        focal_modulation = tf.pow(1 - y_pred, gamma)

        weights = tf.reduce_sum(class_weights_tensor * y_true_oh, axis=-1)
        loss_map = weights * tf.reduce_sum(focal_modulation * cross_entropy, axis=-1)

        return tf.reduce_mean(loss_map)

    return loss
