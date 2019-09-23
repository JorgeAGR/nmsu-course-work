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

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 16

train_data = concentGauss(500, 5, 1, 1, seed=5)
inner = np.where(train_data[:,2] == 1)
outer = np.where(train_data[:,2] == -1)

knn = KNN(train_data[:,:2], train_data[:,2])

delta = 0.05
x_grid = np.arange(-5, 5+delta, delta)
y_grid = np.arange(-5, 5+delta, delta)

xx, yy = np.meshgrid(x_grid, y_grid)

k_neighbors = [1, 5, 10]

fig = plt.figure()
ax = [plt.subplot2grid((4,4), (0,0), colspan=2, rowspan = 2, fig=fig),
      plt.subplot2grid((4,4), (0,2), colspan=2, rowspan = 2, fig=fig),
      plt.subplot2grid((4,4), (2,0), colspan=2, rowspan = 2, fig=fig),
      plt.subplot2grid((4,4), (2,2), colspan=2, rowspan = 2, fig=fig)]

for n, k in enumerate(k_neighbors):
    pred_surface = np.zeros_like(xx)
    for i in range(len(xx)):
        data = np.hstack((xx[i].reshape(len(xx[i]), 1), yy[i].reshape(len(yy[i]), 1)))
        pred_surface[i] = knn.predict(data, k)
    
    ax[n+1].contourf(xx, yy, pred_surface, colors=('lightgreen', 'blue'))
    ax[n+1].set_xlim(-5, 5)
    ax[n+1].set_ylim(-5, 5)
    ax[n+1].set_title(r'$K = $' + str(k))
    ax[n+1].set_xlabel(r'$x_1$')
    ax[n+1].set_ylabel(r'$x_2$')
    ax[n+1].xaxis.set_major_locator(mtick.MultipleLocator(5))
    ax[n+1].xaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax[n+1].yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax[n+1].yaxis.set_minor_locator(mtick.MultipleLocator(1))
    
ax[0].scatter(train_data[inner][:,0], train_data[inner][:,1], 100, marker='+', color='blue', label=r'target: $+1$')
ax[0].scatter(train_data[outer][:,0], train_data[outer][:,1], 100, marker='x', color='lightgreen', label=r'target: $-1$')
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)
ax[0].set_xlabel(r'$x_1$')
ax[0].set_ylabel(r'$x_2$')
ax[0].xaxis.set_major_locator(mtick.MultipleLocator(5))
ax[0].xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax[0].yaxis.set_major_locator(mtick.MultipleLocator(5))
ax[0].yaxis.set_minor_locator(mtick.MultipleLocator(1))
fig.tight_layout(w_pad=0.8, h_pad=0.8)
plt.savefig('../prob6a.eps', dpi=500)