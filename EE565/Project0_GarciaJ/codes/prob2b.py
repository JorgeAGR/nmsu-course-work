#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:41:57 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from requiredFunctions.doublemoon import doublemoon

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

N = 500
w = 0.6
r = 1
d = 0

dist = doublemoon(N, w, r, d, seed=1)
upper_moon = dist[np.where(dist[:,1] > 0)]
lower_moon = dist[np.where(dist[:,1] < 0)]

fig, ax = plt.subplots()
ax.scatter(upper_moon[:,0], upper_moon[:,1], 100, marker='+', color='blue', label=r'target: $+$1')
ax.scatter(lower_moon[:,0], lower_moon[:,1], 100, marker='x', color='lightgreen', label=r'target: $-$1')
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.set_xlabel(r'x$_1$')
ax.set_ylabel(r'x$_2$')
ax.legend()
plt.tight_layout()
plt.savefig('../prob2.eps', dpi=500)