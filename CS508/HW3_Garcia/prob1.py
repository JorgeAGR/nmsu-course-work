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

splits = 5
kf = KFold(n_splits=splits, random_state=0)
kf.get_n_splits(x)

k_neighbors = 10
train_score = np.zeros(splits)
test_score = np.zeros(splits)
i = 0
for train_ind, test_ind in kf.split(x):
    x_train, y_train = x[train_ind], y[train_ind]
    x_test, y_test = x[test_ind], y[test_ind]
    
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(x_train, y_train)
    train_score[i] = knn.score(x_train, y_train)
    test_score[i] = knn.score(x_test, y_test)
    i += 1