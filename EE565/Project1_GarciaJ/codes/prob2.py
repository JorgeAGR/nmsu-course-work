#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:21:34 2019

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

N = 500

data1 = circGauss(N//2, 3, 0, 0, seed=1)
data2 = circGauss(N//2, 3, 5, 5, seed=2)
data = np.vstack((data1, data2))

log_learning_rates = np.arange(-3, -1.1, 0.1)

epochs_converge = np.zeros(log_learning_rates.shape[0])
epochs_converge_std = np.zeros(log_learning_rates.shape[0])
centroids = []
for i, log_eta in enumerate(log_learning_rates):
    trials = 10
    epochs = np.zeros(trials)
    for j in range(trials):
        kmeans = KMeans()
        c, e = kmeans.fit_online(data, 2, learn_rate=10**log_eta, converge=1e-1, max_epochs=20)
        epochs[j] += e
    epochs_converge[i] += epochs.mean()
    epochs_converge_std[i] += epochs.std()
    centroids.append(c)
    
fig, ax = plt.subplots()
ax.errorbar(log_learning_rates, epochs_converge, yerr=epochs_converge_std, color='black', capsize=5)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.set_xlim(-3.1, -1.1)
ax.set_ylim(0, 24)
ax.set_ylabel(r'Epochs')
ax.set_xlabel(r'$\log\eta$')
plt.tight_layout()
plt.savefig('../prob2.eps', dpi=500)