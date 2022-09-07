from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt


## TODO: custon color vector instead of random, add legend with cluster names (labels)
def plot_clustering(latent, clusters):

    clusters = clusters[:latent.shape[0]] # because of weird batch_size

    hex_colors = []
    color_map = {}
    for label in np.unique(clusters):
        hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
        color_map[label] = hex_colors[-1]

    colors = [hex_colors[int(i)] for i in clusters]

    latent_pca = TruncatedSVD(n_components=2).fit_transform(latent)
    latent_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(latent)

    fig, axs = plt.subplots(2, figsize=(10,20),sharey=False,sharex=False)

    axs[0].scatter(latent_pca[:, 0], latent_pca[:, 1], c=colors, marker='*', linewidths=0)
    axs[0].set_title('PCA')

    axs[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=colors, marker='*', linewidths=0)
    axs[1].set_title('tSNE')

    fig.show()

    return hex_colors , color_map

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
