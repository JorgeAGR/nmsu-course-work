#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:49:59 2019

@author: jorgeagr
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from requiredFunctions.doubleMoon import doubleMoon
import matplotlib as mpl
import matplotlib.pyplot as plt

width = 10
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

N = 1000
r = 1
w = 0.6
d_vals = [0.5, -0.5]
titles = ['b', 'c']
for i, d in enumerate(d_vals):
    # Part B
    data = doubleMoon(N, w, r, d, seed=0)
    x_train = data[:,:2]
    y_train = data[:,-1]
    
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1)
    tree.fit(x_train, y_train)
    
    fig, ax = plt.subplots()
    plot_tree(tree, ax=ax)
    fig.savefig('../prob1' + titles[i] + '.eps')