# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:16:38 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.kNearestNeighbors import KNN
from requiredFunctions.doubleMoon import doubleMoon
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

train_data = doubleMoon(500, 0.6, 1, 0)

knn = KNN(train_data[:,:2], train_data[:,2])

delta = 0.01
x_grid = np.arange(-1.5, 2.5+delta, delta)
y_grid = np.arange(-1.5, 1.5+delta, delta)

xx, yy = np.meshgrid(x_grid, y_grid)

k_neighbors = [1, 5, 10]

for k in k_neighbors:
    pred_surface = np.zeros_like(xx)
    for i in range(len(xx)):
        data = np.hstack((xx[i].reshape(len(xx[i]), 1), yy[i].reshape(len(yy[i]), 1)))
        pred_surface[i] = knn.predict(data, k)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, pred_surface, alpha=0.8, colors=('lightgreen', 'blue'))
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 1.5)