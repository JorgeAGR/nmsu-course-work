# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:31 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

data = pd.read_csv('data.csv')

cluster_sizes = np.arange(2, 21, 1)
silh_per_clusters = np.zeros((len(cluster_sizes),2))
for k_clusters in cluster_sizes:
    trials = 10
    silh_coeffs = np.zeros(trials)
    for i in range(trials):
        kmeans = KMeans(n_clusters=k_clusters, init='random', n_init=1, random_state=i)
        kmeans.fit(data.values)
        silh_coeffs[i] = silhouette_score(data.values, kmeans.labels_)
    silh_per_clusters[k_clusters-2,0] = silh_coeffs.mean()
    silh_per_clusters[k_clusters-2,1] = silh_coeffs.std()
    
fig, ax = plt.subplots(nrows=2)
ax[0].errorbar(cluster_sizes, silh_per_clusters[:,0], yerr=silh_per_clusters[:,1], capsize=5,
               marker='o', markersize=5, markeredgewidth=2, color='black')