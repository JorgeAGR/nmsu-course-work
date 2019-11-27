#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:41:09 2019

@author: jorgeagr
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve
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
lof = LocalOutlierFactor(n_neighbors=10, metric='euclidean')
labels = lof.fit_predict(data.values[:,:-1])
score = lof.negative_outlier_factor_

# ROC curve for the malignant label
fpr, tpr, tresholds = roc_curve(data.loc[:,10].values, score, -1)

# Plotting routine for ROC
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='Malignant ROC')
ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='black', label='Chance')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.legend()
#plt.savefig('figs/prob2_logreg_roc.eps', dpi=500)