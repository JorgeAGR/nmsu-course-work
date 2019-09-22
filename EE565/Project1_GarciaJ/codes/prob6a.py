# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:11:08 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.kNearestNeighbors import KNN
from requiredFunctions.concentGauss import concentGauss
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

train_data = concentGauss(500, 5, 1, 1, seed=5)

knn = KNN(train_data[:,:2], train_data[:,2])

delta = 0.1
x_grid = np.arange(-5, 5+delta, delta)
y_grid = np.arange(-5, 5+delta, delta)

extra_x_left, extra_x_right = np.arange(-10, -6, 1), np.arange(5, 11, 1)
extra_y_below, extra_y_above = np.arange(-10, -6, 1), np.arange(5, 11, 1)

x_grid = np.hstack((extra_x_left, x_grid, extra_x_right))
y_grid = np.hstack((extra_y_below, y_grid, extra_y_above))

xx, yy = np.meshgrid(x_grid, y_grid)

k_neighbors = [1, 5, 10]

for k in k_neighbors:
    pred_surface = np.zeros_like(xx)
    for i in range(len(xx)):
        data = np.hstack((xx[i].reshape(len(xx[i]), 1), yy[i].reshape(len(yy[i]), 1)))
        pred_surface[i] = knn.predict(data, k)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, pred_surface, alpha=0.8, colors=('lightgreen', 'blue'))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)