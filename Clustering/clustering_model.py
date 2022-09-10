import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import numpy as np
import sys
from keras import losses
from keras import regularizers
import keras.backend as K

#tf.keras.losses.CosineSimilarity(axis=-1, reduction="auto", name="cosine_similarity")
#tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
#tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")

# model.add(layers.Dense(8,activation = tf.keras.activations.relu, input_shape=(8,),
#                        kernel_regularizer=regularizers.l2(0.01),
#                        activity_regularizer=regularizers.l1(0.01)))

# def custom_loss(y_true, y_pred):
#     return K.mean(y_true - y_pred)**2

class MeanSquaredError(losses.Loss):

  def call(self, y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
