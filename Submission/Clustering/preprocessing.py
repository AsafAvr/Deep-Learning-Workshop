from time import time
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os


def extract_speed_rot(traj):
    x_mod = traj.copy()
    for j in range(traj.shape[0]):
        for i in range(traj.shape[1]-1):
            x_mod[j,i,0] = (np.sqrt( (traj[j,i+1,1]-traj[j,i,1])**2 + (traj[j,i+1,0]-traj[j,i,0])**2 ))
            x_mod[j,i,1] = np.arctan( (traj[j,i+1,1]-traj[j,i,1])/(traj[j,i+1,0]-traj[j,i,0]) )
        x_mod[j,-1,:] = x_mod[j,-2,:]
    return x_mod

def normalize_axis(X_train):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    return x_norm , scaler

def idx_no_outliers_after_norm(x_norm):
    x_norm_re = x_norm.copy()
    x_norm_re = x_norm.reshape(x_norm.shape[0], x_norm.shape[1]*x_norm.shape[2])
    idx = np.where(np.all(np.abs(x_norm_re)<8, axis =1))[0]
    return idx