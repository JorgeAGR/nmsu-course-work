# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:20:30 2019

@author: jorge
"""

import numpy as np
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

file_dir = 'data/'
files = ['kmeans.npy', 'kmeans++.npy', 'agg_clust.npy', 'gaussian_mixture.npy']
labels = ['K-Means', 'K-Means++', 'Agg. Clustering', 'Gaussian Mixture']

clusters = np.arange(2,21,1)
fig, ax = plt.subplots()
for i, f in enumerate(files):
    silh = np.load(file_dir + f)
    if i != 2:
        avg = silh[:,0]
        std = silh[:,1]
        ax.errorbar(clusters, avg, yerr=std, capsize=5, marker='o',
                    markersize=5, markeredgewidth=2, label=labels[i])
    else:
        ax.plot(clusters, silh, marker='o', markersize=5, markeredgewidth=2,
                label=labels[i])
ax.set_ylim(0, 0.2)
ax.set_xlim(1, 21)
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.01))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('figs/prob6.eps', dpi=500)