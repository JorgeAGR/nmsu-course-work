# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:58:32 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
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

# Load data and replace class string with integer
data = pd.read_csv('data/iris.csv', header=None, index_col=None)
data.replace('Iris-setosa', 0, inplace=True)
data.replace('Iris-versicolor', 1, inplace=True)
data.replace('Iris-virginica', 2, inplace=True)

x = data.loc[:, :3].values
y = data.loc[:, 4].values

# Define 5 folds to split the data set
splits = 5
kf = KFold(n_splits=splits, random_state=0)
kf.get_n_splits(x)

# K neighbors to iterate and performance arrays
k_neighbors = np.arange(1, 51, 1)
train_avg = np.zeros(len(k_neighbors))
train_std = np.zeros(len(k_neighbors))
test_avg = np.zeros(len(k_neighbors))
test_std = np.zeros((2,len(k_neighbors)))
for k in k_neighbors:
    # Specific instance of k-neighbors 5-fold scores
    train_score = np.zeros(splits)
    test_score = np.zeros(splits)
    i = 0
    for train_ind, test_ind in kf.split(x):
        # Split into training and testing, fit and evaluate
        x_train, y_train = x[train_ind], y[train_ind]
        x_test, y_test = x[test_ind], y[test_ind]    
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        train_score[i] = knn.score(x_train, y_train)
        test_score[i] = knn.score(x_test, y_test)
        i += 1
    # Save 5-fold average accuracy and standard deviation
    train_avg[k-1] = train_score.mean()
    train_std[k-1] = train_score.std()
    test_avg[k-1] = test_score.mean()
    test_std[0][k-1] = test_score.std()

# Cap error at 0 or 1, since accuracy is constrained to that range
test_std[1,:][(test_std[0,:] + test_avg) > 1] = 1 - test_avg[(test_std[0,:] + test_avg) > 1]
test_std[1,:][(test_std[0,:] + test_avg) < 1] = test_std[0,:][(test_std[0,:] + test_avg) < 1]
test_std[0,:][(test_avg - test_std[0,:]) < 0] = test_avg[(test_avg - test_std[0,:]) < 0]

# Plot the results
fig, ax = plt.subplots()
ax.errorbar(k_neighbors, test_avg, yerr=test_std, capsize=5, label='Testing',
            color='red', marker='o', markersize=5, markeredgewidth=2)
ax.errorbar(k_neighbors, train_avg, yerr=train_std, capsize=5, label='Training',
            color='blue', marker='o', markersize=5, markeredgewidth=2)
ax.set_xlim(-1, 51)
ax.set_ylim(-0.02, 1.02)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(2))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.set_xlabel('k Neighbors')
ax.set_ylabel('Accuracy')
ax.legend()
fig.tight_layout()
plt.savefig('figs/prob1_kfold_knn.eps', dpi=500)