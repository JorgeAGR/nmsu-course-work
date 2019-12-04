#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:35:28 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.histogram import histogram
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

width = 10
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

data = np.loadtxt('../data/DatasetA.csv')
n_bins = [20, 200, 2000]
lims = [700, 90, 15]

fig, ax = plt.subplots(nrows=3, sharex=True)
for i in range(len(n_bins)):
    counts, bins = histogram(data, n_bins[i])
    ax[i].bar(bins, counts, np.diff(bins)[0], color='black')
    ax[i].set_title(str(n_bins[i]) + ' bins')
    ax[i].yaxis.set_major_locator(mtick.MultipleLocator(200/(10**i)))
    ax[i].yaxis.set_minor_locator(mtick.MultipleLocator(100/(10**i)))
    ax[i].xaxis.set_major_locator(mtick.MultipleLocator(10))
    ax[i].xaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax[i].set_xlim(-25, 40)
    ax[i].set_ylim(0,lims[i])
    ax[i].set_ylabel('Counts')
ax[-1].set_xlabel(r'$x_0$')
fig.tight_layout(pad=0.5)
fig.savefig('../prob3.eps', dpi=500)