from pydoc import stripid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import numpy as np
import sys
from keras import losses
from keras import regularizers
import keras.backend as K

from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape,LSTM,RepeatVector,Dense,TimeDistributed, \
Conv1D,LeakyReLU,MaxPool1D,Bidirectional,UpSampling2D,Input
from keras.models import Model,Sequential
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

#tf.keras.losses.CosineSimilarity(axis=-1, reduction="auto", name="cosine_similarity")
#tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
#tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")

# model.add(layers.Dense(8,activation = tf.keras.activations.relu, input_shape=(8,),
#                        kernel_regularizer=regularizers.l2(0.01),
#                        activity_regularizer=regularizers.l1(0.01)))

# def custom_loss(y_true, y_pred):
#     return K.mean(y_true - y_pred)**2

def ae_conv(input_shape=(4, 4, 4), filters=[32, 64, 8]):
    stride = 2
    ker = 2
    conv_depth = len(filters)-2
    mul = stride**conv_depth
    model = Sequential()
    ## padding????
    if input_shape[0] % 4 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], ker, strides=stride, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], ker, strides=1, padding='same', activation='relu', name='conv2'))

    model.add(Flatten())
    model.add(Dense(units=filters[-1], name='embedding'))
    model.add(Dense(units=64, activation='relu'))

    model.add(Reshape((int(input_shape[0]/mul), int(input_shape[1]/mul), int(filters[2]))))

    model.add(Conv2DTranspose(filters[0], ker, strides=1, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], ker, strides=stride, padding='same', name='deconv1'))
    model.summary()
    return model


## example of loss
class MeanSquaredError(losses.Loss):

  def call(self, y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)


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

def temporal_autoencoder(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.

    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide time series length
        n_units: numbers of units in the two BiLSTM layers

    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    assert(timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)  #decoded = UpSampling1D(pool_size, name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)  #output = Conv1D(1, kernel_size, strides=strides, padding='same', activation='linear', name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, n_units[1]), name='decoder_input')

    # Internal layers in decoder
    decoded = autoencoder.get_layer('reshape')(encoded_input)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder
