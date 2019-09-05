#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:56:20 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from requiredFunctions.concentGauss import concentGauss

width = 10
height = 10

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

dist = concentGauss(500, 5, 1, 1, seed=10)
inner_gauss = dist[np.where(dist[:,2] == 1)]
outer_gauss = dist[np.where(dist[:,2] == -1)]

fig, ax = plt.subplots()
ax.scatter(inner_gauss[:,0], inner_gauss[:,1], 100, marker='+', color='blue', label=r'target: $+$1')
ax.scatter(outer_gauss[:,0], outer_gauss[:,1], 100, marker='x', color='lightgreen', label=r'target: $-$1')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax.set_xlabel(r'x$_1$')
ax.set_ylabel(r'x$_2$')
ax.legend()
plt.tight_layout()
plt.savefig('../prob3.eps', dpi=500)