# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:55:45 2019

@author: jorge
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

train_data = np.loadtxt('../data/curvefitting.txt')
sinfunc = lambda x: np.sin(2*np.pi*x)
x_grid = np.linspace(0, 1, num=200)

m_power = 9
ln_reg = [-18, 0]
for i, ln_lambda in enumerate(ln_reg):
    
    polyfit = PolyFit(m_power)
    polyfit.fit_LS(train_data[:,0], train_data[:,1], reg = np.exp(ln_lambda))
    print('ln lambda =', ln_lambda, 'Regularized Fit Weights:')
    print(polyfit.weights.flatten(), end='\n\n')
    
    fig, ax = plt.subplots()
    ax.scatter(train_data[:,0], train_data[:,1], 100, facecolors='none', 
           edgecolors='blue', linewidth=2)
    ax.plot(x_grid, polyfit.predict(x_grid), color='red')
    ax.plot(x_grid, sinfunc(x_grid), color='lightgreen')
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-0.1, 1.1)
