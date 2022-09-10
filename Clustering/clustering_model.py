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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()

    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))


class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()
