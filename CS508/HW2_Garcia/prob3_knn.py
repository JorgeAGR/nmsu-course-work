# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:24:24 2019

@author: jorge

SATURDAY 1 - 6 pm @ mesilla
SUNDAY 1- 6 pm
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

# Load train data. Split into X matrix and Y array
train_data = pd.read_csv('data/glass_train.csv', index_col=0)
train_x = train_data.iloc[:,:-1].values
train_y = train_data.iloc[:,-1].values

# Load test data. Split into X matrix and Y array.
test_data = pd.read_csv('data/glass_test.csv', index_col=0)
test_x = test_data.iloc[:,:-1].values
test_y = test_data.iloc[:,-1].values

# Define array of range of k-neighbors to consider
k_range = [i for i in range(1,11)]
k_range.append(15), k_range.append(20), k_range.append(25)

# Function to evaluate k-Nearest Neighbors performance for different metrics and
# k-neighbors considered.
def evaluate_neighbors(metric):
    train_acc = np.zeros(len(k_range))
    test_acc = np.zeros(len(k_range))
    
    for i, k in enumerate(k_range):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(train_x, train_y)
        train_acc[i] = knn.score(train_x, train_y)
        test_acc[i] = knn.score(test_x, test_y)
    
    return train_acc, test_acc

# Different metrics to consider for the KNN algorithm
metrics = ['euclidean', 'manhattan', 'cosine']

# Evaluate KNN accuracy as a function of metric and k-neighbors
# Plot accuracy results
for metric in metrics:
    train_acc, test_acc = evaluate_neighbors(metric)
    
    fig, ax = plt.subplots()
    ax.plot(k_range, train_acc, color='blue', label='Training',
            marker='o', markersize=5, markeredgewidth=2)
    ax.plot(k_range, test_acc, color='red', label='Testing',
            marker='o', markersize=5, markeredgewidth=2)
    ax.set_xlim(0, 26)
    ax.set_ylim(0.5, 1.02)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
    ax.set_xlabel('k-Neighbors')
    ax.set_ylabel('Accuracy')
    ax.set_title(metric)
    ax.legend()
    fig.tight_layout()
    plt.savefig('figs/prob3_' + metric + '.eps', dpi=500)