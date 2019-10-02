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
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/glass_train.csv', index_col=0)
train_x = train_data.iloc[:,:9].values
train_y = train_data.iloc[:,-1].values

test_data = pd.read_csv('data/glass_test.csv', index_col=0)
test_x = test_data.iloc[:,:9].values
test_y = test_data.iloc[:,-1].values

k_range = [i for i in range(1,11)]
k_range.append(15), k_range.append(20), k_range.append(25)

def evaluate_neighbors(metric):
    train_acc = np.zeros(len(k_range))
    test_acc = np.zeros(len(k_range))
    
    for i, k in enumerate(k_range):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(train_x, train_y)
        train_acc[i] = knn.score(train_x, train_y)
        test_acc[i] = knn.score(test_x, test_y)
    
    return train_acc, test_acc

metrics = ['euclidean', 'manhattan', 'cosine']

for metric in metrics:
    train_acc, test_acc = evaluate_neighbors(metric)
    
    fig, ax = plt.subplots()
    ax.plot(k_range, train_acc)
    ax.plot(k_range, test_acc)