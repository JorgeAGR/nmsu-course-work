#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:51:21 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kMeans import KMeans
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

kmeans = KMeans()
k_max = 20
k_groups = np.arange(2, k_max+1, 1)

cost_per_k = np.zeros(len(k_groups))
cost_per_k_std = np.zeros(len(k_groups))
for i, k in enumerate(k_groups):
    trials = 100
    cost = np.zeros(trials)
    for j in range(trials):
        kmeans.fit_batch(data, k)
        cost[j] += kmeans.eval_cost(data)
    cost_per_k[i] += cost.mean()
    cost_per_k_std[i] += cost.std()
    
fig, ax = plt.subplots()
ax.errorbar(k_groups, cost_per_k, yerr=cost_per_k_std, color='black')
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_xlim(0, 21)
ax.set_ylim(0, 350)
ax.set_ylabel(r'Cost')
ax.set_xlabel(r'K')
plt.tight_layout()
plt.savefig('../prob1d.eps', dpi=500)