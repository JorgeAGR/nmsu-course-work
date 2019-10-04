# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:08:26 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.leastSquaresClassifier import LeastSquares_Classifier
from requiredFunctions.oneHotEncode import oneHotEncode
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

train_x = np.loadtxt('../data/DatasetA_data.csv', delimiter=',')
train_y = np.loadtxt('../data/DatasetA_labels.csv', delimiter=',')

ls_class = LeastSquares_Classifier(train_x, train_y)

x0_min, x0_max = train_x[:,0].min() - 1, train_x[:,0].max() + 1
x1_min, x1_max = train_x[:,1].min() - 1, train_x[:,1].max() + 1

xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
                       np.arange(x1_min, x1_max, 0.01))

cc = ls_class.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

pred_surface = np.zeros_like(xx0)
for i in range(len(xx0)):
    point = np.hstack((xx0[i].reshape(len(xx0[i]), 1), xx1[i].reshape(len(xx1[i]), 1)))
    pred_surface[i] = ls_class.predict(point)

cmap = plt.get_cmap('Set1')
plot_colors = mpl.colors.ListedColormap(cmap(np.arange(0, 2, 1)))
fig, ax = plt.subplots()
ax.contourf(xx0, xx1, cc, cmap=plot_colors, alpha=0.5)
ax.scatter(train_x[:,0], train_x[:,1], c=train_y, cmap=plot_colors)