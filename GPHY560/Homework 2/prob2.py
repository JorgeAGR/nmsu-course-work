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

# Problem 2
np.random.seed(0)
samples = 10000
chisq_50dof = stats.chi2.rvs(50, size=samples)
chisq_100dof = stats.chi2.rvs(100, size=samples)
f_approx = (chisq_50dof/50) / (chisq_100dof/100)
x_grid = np.linspace(f_approx.min(), f_approx.max(), num=200)
f_theory = stats.f.pdf(x_grid, 50, 100)

fig, ax = plt.subplots()
ax.hist(f_approx, 100, align='left',
        density=True, histtype='stepfilled', linewidth=2, color='black', label='Sampled Distribution')
ax.plot(x_grid, f_theory, linewidth=4, color='red', label=r'$F$' + ' Distribution')
ax.set_xlim(0.25, 2.25)
ax.set_ylim(0, 2)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$pdf(x)$')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob2.eps', dpi=500)