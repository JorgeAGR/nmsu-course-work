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

log_learning_rates = np.arange(-3, 0.1, 0.1)

epochs_converge = np.zeros(log_learning_rates.shape[0])
epochs_converge_std = np.zeros(log_learning_rates.shape[0])
centroids = []
for i, log_eta in enumerate(log_learning_rates):
    trials = 10
    epochs = np.zeros(trials)
    for j in range(trials):
        kmeans = KMeans()
        c, e = kmeans.fit_online(data, 2, learn_rate=10**log_eta, converge=1e-1, max_epochs=30)
        epochs[j] += e
    epochs_converge[i] += epochs.mean()
    epochs_converge_std[i] += epochs.std()
    centroids.append(c)
    
fig, ax = plt.subplots()
ax.errorbar(log_learning_rates, epochs_converge, yerr=epochs_converge_std)