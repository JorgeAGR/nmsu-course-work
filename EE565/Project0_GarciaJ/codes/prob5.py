#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:57:20 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from requiredFunctions.noisySin import noisySin

golden_ratio = (np.sqrt(5) + 1) / 2
width = 15
height = width / golden_ratio

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

N = 50
var = 0.05
data = noisySin(N, var)
x_grid = np.linspace(0, 1, num=500)

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], 100, facecolors='none', 
           edgecolors='blue', label='Data')
ax.plot(x_grid, np.sin(2*np.pi*x_grid), color='lightgreen',
        label='Function')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-1.5, 1.5)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.legend()
plt.tight_layout()

file_data = np.loadtxt('../data/curvefitting.txt')
fig2, ax2 = plt.subplots()
ax2.scatter(file_data[:,0], file_data[:,1], 100, facecolors='none', 
           edgecolors='blue', label='Data')
ax2.plot(x_grid, np.sin(2*np.pi*x_grid), color='lightgreen',
        label='Function')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-1.5, 1.5)
ax2.xaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax2.xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax2.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.legend()
plt.tight_layout()