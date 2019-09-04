#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:06:49 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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

data = np.loadtxt('../data/faithful.txt')

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], 100, facecolors='none', 
           edgecolors='lightgreen', linewidth=2,
           label='Data')
ax.set_xlim(1, 6)
ax.set_ylim(40, 100)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))
ax.set_xlabel('Eruption Time [min]')
ax.set_ylabel('Waiting Time [min]')
ax.legend()
plt.tight_layout()
plt.savefig('../prob6.eps', dpi=500)