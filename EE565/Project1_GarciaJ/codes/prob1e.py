#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:36:21 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kmeans import KMeans
#from requiredFunctions.color_map import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

N = 50

data1 = circGauss(N//2, 3, 0, 0, seed=1)
data2 = circGauss(N//2, 3, 5, 5, seed=2)
data = np.vstack((data1, data2))

data_ind = np.arange(0, N, 1)
np.random.seed(seed=7)
np.random.shuffle(data_ind)

centroids = [data[data_ind[:5]], data[data_ind[5:10]], data[data_ind[10:15]]]

for cent in centroids:
    kmeans = KMeans()
    new_cent, _, _ = kmeans.fit_batch(data, 5, centroids=cent)
    pred_class = kmeans.predict_cluster(data)
    
    cmap = plt.get_cmap('Set1')
    plot_colors = mpl.colors.ListedColormap(cmap(np.arange(0, 5, 1)))
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1],
               c=pred_class, cmap=plot_colors)
    ax.scatter(cent[:,0], cent[:,1], color='orange', marker='x')
    ax.scatter(new_cent[:,0], new_cent[:,1], color='red', marker='x')
    for i in range(5):
        ax.plot([cent[i,0], new_cent[i,0]], [cent[i,1], new_cent[i,1]], 
                color='black')