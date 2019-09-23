#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:27:54 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.polyFit import PolyFit
from requiredFunctions.noisySin import noisySin
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
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

train_data = np.loadtxt('../data/curvefitting.txt')
test_data = noisySin(50, 0.3**2, seed=5)

sinfunc = lambda x: np.sin(2*np.pi*x)

m_powers = np.arange(0, 10, 1)
x_grid = np.linspace(0, 1, num=200)

# Part A
p = 0
train_error = np.zeros(len(m_powers))
test_error = np.zeros(len(m_powers))
fig, ax = plt.subplots(nrows=2, ncols=2)
for m in m_powers:
    polyfit = PolyFit(m)
    _, tr_err = polyfit.fit_LS(train_data[:,0], train_data[:,1])
    train_error[m] += np.sqrt(tr_err / len(train_data))
    _, te_err = polyfit.test(test_data[:,0], test_data[:,1])
    test_error[m] += np.sqrt(te_err / len(test_data))
    
    if m in [0, 1, 3, 9]:
        print(m, 'Order Fit Weights:')
        print(polyfit.weights.flatten(), end='\n\n')
        ax[p//2][p%2].scatter(train_data[:,0], train_data[:,1], 100, facecolors='none', 
               edgecolors='blue', linewidth=2)
        
        ax[p//2][p%2].plot(x_grid, polyfit.predict(x_grid), color='red')
        ax[p//2][p%2].plot(x_grid, sinfunc(x_grid), color='lightgreen')
        ax[p//2][p%2].xaxis.set_major_locator(mtick.MultipleLocator(1))
        ax[p//2][p%2].yaxis.set_major_locator(mtick.MultipleLocator(1))
        ax[p//2][p%2].set_xlabel(r'$x$')
        ax[p//2][p%2].set_ylabel(r'$t$')
        ax[p//2][p%2].text(0.6, 1, 'M = ' + str(m))
        ax[p//2][p%2].set_ylim(-1.5, 1.5)
        ax[p//2][p%2].set_xlim(-0.1, 1.1)
        p += 1
fig.tight_layout()
plt.savefig('../prob4a.eps', dpi=500)

# Part B
fig_err, ax_err = plt.subplots()
ax_err.plot(m_powers, train_error, '-o', color='blue', label='Training',
            markersize=10, markeredgewidth=2, fillstyle='none')
ax_err.plot(m_powers, test_error, '-o', color='red', label='Testing',
            markersize=10, markeredgewidth=2, fillstyle='none')
ax_err.xaxis.set_major_locator(mtick.MultipleLocator(3))
ax_err.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax_err.set_xlim(-1, 10)
ax_err.set_ylim(0, 1)
ax_err.set_xlabel(r'$M$')
ax_err.set_ylabel(r'$E_\mathrm{RMS}$')
ax_err.legend()
fig_err.tight_layout()
plt.savefig('../prob4b.eps', dpi=500)