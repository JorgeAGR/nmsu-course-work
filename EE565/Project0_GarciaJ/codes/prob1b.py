#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:21:15 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from requiredFunctions.circGauss import circGauss

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

N = 500
var = 3

dist_1 = circGauss(N, var, 0, 0)

dist_2 = circGauss(N, var, 5, 5)

fig, ax = plt.subplots()
ax.scatter(dist_1[:,0], dist_1[:,1], color='black', marker='.')
ax.scatter(dist_2[:,0], dist_2[:,1], color='black', marker='.')
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
ax.set_xlabel(r'x$_1$')
ax.set_ylabel(r'x$_2$')
plt.tight_layout()