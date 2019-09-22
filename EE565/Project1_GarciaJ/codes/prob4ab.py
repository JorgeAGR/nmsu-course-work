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
mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

'''
NOT FUCKING WORKING WHYYYYYY

'''

train_data = np.loadtxt('../data/curvefitting.txt')
test_data = noisySin(50, 0, seed=5)

sinfunc = lambda x: np.sin(2*np.pi*x)

m_powers = np.arange(0, 10, 1)
x_grid = np.linspace(0, 1, num=200)

train_error = np.zeros(len(m_powers))
test_error = np.zeros(len(m_powers))
for m in m_powers:
    polyfit = PolyFit(m)
    _, tr_err = polyfit.fit_LS(train_data[:,0], train_data[:,1])
    train_error[m] += np.sqrt(tr_err / len(train_data))
    pred, te_err = polyfit.test(test_data[:,0], test_data[:,1])
    test_error[m] += np.sqrt(te_err / len(test_data))
    
    print(m, 'err', np.abs(pred-test_data[:,1]).sum())
    
    if m in [0, 1, 3, 9]:
        print(m, 'Order Fit Weights:')
        print(polyfit.weights.flatten(), end='\n\n')
        fig, ax = plt.subplots()
        ax.scatter(train_data[:,0], train_data[:,1], 100, facecolors='none', 
               edgecolors='blue', linewidth=2)
        
        ax.scatter(test_data[:,0], test_data[:,1], 50, color='orange')
        ax.scatter(test_data[:,0], pred, 50, color='red')
        
        ax.plot(x_grid, polyfit.predict(x_grid), color='red')
        ax.plot(x_grid, sinfunc(x_grid), color='lightgreen')
        ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
        ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-0.1, 1.1)

fig_err, ax_err = plt.subplots()
ax_err.plot(m_powers, train_error)
ax_err.plot(m_powers, test_error)