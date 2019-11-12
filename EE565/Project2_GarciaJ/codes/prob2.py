# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 00:39:07 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.leastSquaresClassifier import LeastSquares_Classifier
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

train_x = np.loadtxt('../data/DatasetB_data.csv', delimiter=',')
train_y = np.loadtxt('../data/DatasetB_labels.csv', delimiter=',')

ls_class = LeastSquares_Classifier(train_x, train_y)
accuracy = ls_class.score(train_x, train_y)
print('Dataset B Accuracy:', accuracy)

x0_min, x0_max = np.floor(train_x[:,0].min()), np.ceil(train_x[:,0].max())
x1_min, x1_max = np.floor(train_x[:,1].min()), np.ceil(train_x[:,1].max())

xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
                       np.arange(x1_min, x1_max, 0.01))

cc = ls_class.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

cmap = plt.get_cmap('tab20')
cmap_scatter = mpl.colors.ListedColormap(cmap((0, 6, 4)))
cmap_contour = mpl.colors.ListedColormap(cmap((1, 7, 5)))

fig, ax = plt.subplots()
ax.contourf(xx0, xx1, cc, cmap=cmap_contour)
ax.scatter(train_x[:,0], train_x[:,1], c=train_y,
           cmap=cmap_scatter, edgecolor='black')
ax.set_xlim(x0_min, x0_max)
ax.set_ylim(x1_min, x1_max)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig.tight_layout()
plt.savefig('../prob2.eps', dpi=500)

plt.show()