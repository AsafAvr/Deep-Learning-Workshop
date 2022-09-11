from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import config

## TODO: custon color vector instead of random, add legend with cluster names (labels)
def plot_clustering(latent, clusters):

    clusters = clusters[:latent.shape[0]] # because of weird batch_size

    colors = [config.color_options[int(i)] for i in clusters]

    latent_pca = TruncatedSVD(n_components=2).fit_transform(latent)
    latent_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(latent)

    fig, axs = plt.subplots(2, figsize=(10,20),sharey=False,sharex=False)

    axs[0].scatter(latent_pca[:, 0], latent_pca[:, 1], c=colors, marker='*', linewidths=0)
    axs[0].set_title('PCA')

    axs[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=colors, marker='*', linewidths=0)
    axs[1].set_title('tSNE')

    fig.show()

def plot_dual_clustering(well1, clusters1, well2, clusters2, hex_colors):

    clusters1 = clusters1[:well1.shape[0]] # because of weird batch_size
    clusters2 = clusters2[:well2.shape[0]] # because of weird batch_size
    colors1 = [hex_colors[int(i)] for i in clusters1]
    colors2 = [hex_colors[int(i)] for i in clusters2]

    pca1 = TruncatedSVD(n_components=2).fit_transform(well1)
    tsne1 = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(well1)

    pca2 = TruncatedSVD(n_components=2).fit_transform(well2)
    tsne2 = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(well2)

    fig, axs = plt.subplots(2, 2, figsize=(20,20),sharey=True,sharex=True)

    axs[0,0].scatter(pca1[:, 0], pca1[:, 1], c=colors1, marker='*', linewidths=0)
    axs[0,0].set_title('PCA on well1')

    axs[1,0].scatter(tsne1[:, 0], tsne1[:, 1], c=colors1, marker='*', linewidths=0)
    axs[1,0].set_title('tSNE on well1')

    axs[0,1].scatter(pca2[:, 0], pca2[:, 1], c=colors2, marker='*', linewidths=0)
    axs[0,1].set_title('PCA on well2')

    axs[1,1].scatter(tsne2[:, 0], tsne2[:, 1], c=colors2, marker='*', linewidths=0)
    axs[1,1].set_title('tSNE on well2')

    fig.show()

def plot_trajectory(kinetic):
    return plt.plot(kinetic[:,0],kinetic[:,1])

def plot_representatives(rep_groups):
    l = len(rep_groups[0])
    L = len(rep_groups)
    fig, axs = plt.subplots(l,L, figsize=(10,10),sharey=True,sharex=True)
    fig.tight_layout()
    for j,reps in enumerate(rep_groups):
        for i,rep in enumerate(reps):
            axs[i,j].plot(rep[:,0],rep[:,1])
    fig.show()

def plot_autoencoder_by_time(orig_list,pred_list , figsize =(10,15)):
    rows = len(orig_list)
    timesteps = orig_list[0].shape[0]
    features = orig_list[0].shape[1]
    fig, axs = plt.subplots(rows,features, figsize=figsize,sharey=True,sharex=True)
    fig.tight_layout()
    for j in range(features):
      for i,_ in enumerate(orig_list):
          data = np.zeros((timesteps,2))
          data[:,0] = pred_list[i][:,j]
          data[:,1] = orig_list[i][:,j]
          idx = [i for i in range(len(data))]
          axs[i,j].plot(idx,data)
    fig.show()

