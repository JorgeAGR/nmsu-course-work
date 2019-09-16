#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:39:17 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kmeans import KMeans
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

N = 500

data1 = circGauss(N//2, 3, 0, 0, seed=1)
data2 = circGauss(N//2, 3, 5, 5, seed=2)
real_centers = np.array([[0, 0], [5, 5]])

sample = np.arange(10, N+2, 2)
avg_error = np.zeros(len(sample))
avg_error_std = np.zeros(len(sample))
for i, n in enumerate(sample):
    data = np.vstack((data1[:n//2], data2[:n//2]))
    trials = 10
    error = np.zeros(trials)
    for j in range(trials):
        kmeans = KMeans()
        kmeans.fit_batch(data, 2)
        error[j] += np.sqrt(((kmeans.centroids - real_centers)**2).sum())
    avg_error[i] += error.mean()
    avg_error_std[i] += error.std()
    
fig, ax = plt.subplots()
ax.errorbar(sample, avg_error, yerr=avg_error_std)
ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))
ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.set_xlim(0, 500)
ax.set_ylim(-0.2, 3.4)
ax.set_ylabel(r'Error')
ax.set_xlabel(r'# of data points')
plt.tight_layout()
plt.savefig('../prob1a.eps', dpi=500)