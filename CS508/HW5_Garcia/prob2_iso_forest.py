#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:34:27 2019

@author: jorgeagr
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

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

# Load data
data = pd.read_csv('data/breast-cancer-wisconsin.csv', header=None)
# Remove ID attribute
data = data.loc[:,data.keys()[1:]]

# NaNs are represented as question marks (?), so convert them into NaNs
for key in data.keys():
    data.loc[:,key] = pd.to_numeric(data.loc[:,key], errors='coerce')
# Drop rows with NaNs
data = data.dropna()
# Remap class attribute to match LOF labels
data.loc[data[10] == 2, 10] = 1
data.loc[data[10] == 4, 10] = -1

# Init and fit LOF
# Outliers are malignant tumors (1 = inlier, -1 = outlier)
forest = IsolationForest(n_estimators=100, max_samples=256)
trials = 10
times = np.zeros(trials)
for i in range(trials):
    tic = time.time()
    forest.fit(data.values[:,:-1])
    toc = time.time()
    times[i] = (toc-tic)*1000 # in ms
print('Fitting time: {:.3f} ms'.format(times.mean()))
# Take negative of scores such that outliers are scored higher, since
# ROC tests score >= threshold
score = -forest.decision_function(data.values[:,:-1])

# ROC curve for the malignant label
fpr, tpr, thresholds = roc_curve(data.loc[:,10].values, score, -1)
auc_val = auc(fpr, tpr)
print('AUC:', auc_val)

# Plotting routine for ROC
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='Malignant ROC')
ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='black', label='Chance')
ax.text(0.8, 0.1, 'AUC = {:.3f}'.format(auc_val), fontweight='bold', fontsize='large')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('figs/prob2_iso_forest.eps', dpi=500)