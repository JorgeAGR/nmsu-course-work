# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:31 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 14
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 14

data = pd.read_csv('data.csv')

cluster_sizes = np.arange(2, 21, 1)
silh_per_clusters = np.zeros(len(cluster_sizes))
for k_clusters in cluster_sizes:
    aggclust = AgglomerativeClustering(n_clusters=k_clusters)
    aggclust.fit(data.values)
    silh_per_clusters[k_clusters-2] = silhouette_score(data.values, aggclust.labels_)
    
best_k = cluster_sizes[np.argmax(silh_per_clusters[:])]
print('Best Cluster Size:', best_k, 'clusters')
best_aggclust = AgglomerativeClustering(n_clusters=best_k)
best_aggclust.fit(data.values)

best_aggclust.cluster_centers_ = np.asarray([data.values[best_aggclust.labels_ == c].mean(axis=0) for c in range(best_k)])

for i, cluster in enumerate(best_aggclust.cluster_centers_):
    print('Cluster', i+1, ':', np.round(cluster,3))

pca = PCA(n_components=2)
x_trans = pca.fit_transform(data.values)

cmap = mpl.cm.tab20
colors = cmap.colors
fig = plt.figure()
ax = [plt.subplot2grid((3,1), (0,0)),
      plt.subplot2grid((3,1), (1,0), rowspan=2)]
ax[0].plot(cluster_sizes, silh_per_clusters, marker='o', markersize=5, markeredgewidth=2,
      color='black', label='Sillhoute Score')
ax[0].legend()
for c in range(best_k):
    ax[1].scatter(x_trans[:,0][best_aggclust.labels_==c], x_trans[:,1][best_aggclust.labels_==c], 
      c=[colors[c]], label=c+1, marker='.')
    fig.tight_layout(pad=0.5)
ax[1].legend()