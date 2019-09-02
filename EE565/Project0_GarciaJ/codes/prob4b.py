#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:08:38 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from requiredFunctions.gaussX import gaussX

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

dist = gaussX(500, 1)
pos_gauss = dist[np.where(dist[:,2] == 1)]
neg_gauss = dist[np.where(dist[:,2] == -1)]

fig, ax = plt.subplots()
ax.scatter(pos_gauss[:,0], pos_gauss[:,1], 100, marker='+', color='blue', label=r'target: $+$1')
ax.scatter(neg_gauss[:,0], neg_gauss[:,1], 100, marker='x', color='lightgreen', label=r'target: $-$1')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.set_xlabel(r'x$_1$')
ax.set_ylabel(r'x$_2$')
ax.legend()
plt.tight_layout()