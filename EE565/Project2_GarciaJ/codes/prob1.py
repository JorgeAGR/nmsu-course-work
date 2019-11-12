# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:08:26 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.leastSquaresClassifier import LeastSquares_Classifier
from requiredFunctions.doubleMoon import doubleMoon
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

data = ['DatasetA', 'DatasetA2', 'double moon']
part = ['a', 'b', 'c']

for p, d in enumerate(data):
    
    if d == 'double moon':
        dubmoon = doubleMoon(5000, 0.6, 1, -0.1, seed=0)
        train_x, train_y = dubmoon[:,:2], dubmoon[:,-1]
    else:
        train_x = np.loadtxt('../data/' + d + '_data.csv', delimiter=',')
        train_y = np.loadtxt('../data/' + d + '_labels.csv', delimiter=',')

    ls_class = LeastSquares_Classifier(train_x, train_y)
    accuracy = ls_class.score(train_x, train_y)
    print(d, 'Accuracy:', accuracy)
    
    x0_min, x0_max = np.floor(train_x[:,0].min()), np.ceil(train_x[:,0].max())
    x1_min, x1_max = np.floor(train_x[:,1].min()), np.ceil(train_x[:,1].max())
    
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
                           np.arange(x1_min, x1_max, 0.01))
    
    cc = ls_class.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
    
    cmap = plt.get_cmap('tab20')
    cmap_scatter = mpl.colors.ListedColormap(cmap((0, 6)))
    cmap_contour = mpl.colors.ListedColormap(cmap((1, 7)))
    
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
    plt.savefig('../prob1'+part[p]+'.eps', dpi=500)
    
    plt.show()