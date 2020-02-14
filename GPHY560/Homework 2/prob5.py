#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:31:17 2020

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

# Problem 5
np.random.seed(0)
trials = 10
samples = 50
rand_norm = np.random.randn(samples, trials)
t_approx = rand_norm.mean(axis=1)

var_theory = 1/np.sqrt(trials)
x_grid = np.linspace(t_approx.min(), t_approx.max(), num=200)
norm_theory = stats.norm.pdf(x_grid, scale=var_theory)
t_theory = stats.t.pdf(x_grid*np.sqrt(trials), trials) * np.sqrt(trials)

fig, ax = plt.subplots()
ax.set_title(r'$t$' + ' Distribution')
ax.hist(t_approx, np.arange(t_approx.min(), t_approx.max()+0.1, 0.1), align='left',
        density=True, histtype='stepfilled', linewidth=2, color='black', label='Sampled Distribution')
ax.plot(x_grid, norm_theory, linewidth=4, color='red', label='Normal Distribution')
ax.plot(x_grid, t_theory, linewidth=4, color='blue', label= r'$t$' + ' Distribution')
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(0, 1.7)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$pdf(x)$')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob5.eps', dpi=500)