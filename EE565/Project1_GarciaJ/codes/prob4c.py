# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:19:30 2019

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


sinfunc = lambda x: np.sin(2*np.pi*x)
x_grid = np.linspace(0, 1, num=200)

m_power = 9
data_samples = [15, 100]

for i, N in enumerate(data_samples):
    train_data = noisySin(N, 0.3**2, seed=15)
    train_data[0,0], train_data[0,1] = 0, 0
    train_data[-1,0], train_data[-1,1] = 1, 0
    polyfit = PolyFit(m_power)
    polyfit.fit_LS(train_data[:,0], train_data[:,1])
    
    fig, ax = plt.subplots()
    ax.scatter(train_data[:,0], train_data[:,1], 100, facecolors='none', 
           edgecolors='blue', linewidth=2)
    ax.plot(x_grid, polyfit.predict(x_grid), color='red')
    ax.plot(x_grid, sinfunc(x_grid), color='lightgreen')
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-0.1, 1.1)
