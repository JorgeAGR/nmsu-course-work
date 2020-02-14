#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:31:16 2020

@author: jorgeagr
"""

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Setting plotting variables
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

# Problem 3
np.random.seed(0)
thresh = 0.4
trials = 100
samples = 50
uniform = np.random.rand(trials, samples)
binomial_approx = (uniform < thresh).sum(axis=0)
x_grid = np.arange(trials+1)
binomial_theory = stats.binom.pmf(x_grid, trials, thresh)

fig, ax = plt.subplots()
ax.hist(binomial_approx, np.arange(binomial_approx.min(), binomial_approx.max()+1, 1), align='left',
        density=True, histtype='stepfilled', linewidth=2, color='black', label='Sampled Distribution')
ax.vlines(x_grid, 0, binomial_theory, linewidth=2, color='red', label='Binomial Distribution')
ax.set_xlim(20, 60)
ax.set_ylim(0, 0.15)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$pdf(x)$')
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.01))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob3.eps', dpi=500)