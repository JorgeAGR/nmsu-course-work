#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:27:54 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.polyFit import PolyFit
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.image as mpimg

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

train_data = np.loadtxt('../data/curvefitting.txt')

sinfunc = lambda x: np.sin(2*np.pi*x)

m_powers = np.arange(0, 10, 1)
x_grid = np.linspace(0, 1, num=200)

train_error = np.zeros(len(m_powers))
test_error = np.zeros(len(m_powers))
for i, m in enumerate(m_powers):
    polyfit = PolyFit(m)
    _, err = polyfit.fit_LS(train_data[:,0], train_data[:,1])
    train_error[i] += np.sqrt(err / len(train_data))
    
    if m in [0, 1, 3, 9]:
        print(polyfit.weights)
        fig, ax = plt.subplots()
        ax.scatter(train_data[:,0], train_data[:,1], 100, facecolors='none', 
               edgecolors='blue', linewidth=2)
        ax.plot(x_grid, polyfit.predict(x_grid), color='red')
        ax.plot(x_grid, sinfunc(x_grid), color='lightgreen')
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-0.1, 1.1)

fig_err, ax_err = plt.subplots()
ax_err.plot(m_powers, train_error)