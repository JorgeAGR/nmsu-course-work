#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:36:21 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kMeans import KMeans
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

j = 0
for cent in centroids:
    kmeans = KMeans()
    new_cent, _ = kmeans.fit_batch(data, 5, centroids=cent)
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
    ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
    ax.set_xlim(-2.5, 8)
    ax.set_ylim(-4.5, 9.5)
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_1$')
    plt.tight_layout()
    j += 1
    plt.savefig('../prob1e_' + str(j) + '.eps', dpi=500)