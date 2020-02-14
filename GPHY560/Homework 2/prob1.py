#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:30:41 2020

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

# Problem 1
np.random.seed(0)
dof = 100
samples = 10000
rand_norm = np.random.randn(dof,samples)
chisq_approx = (rand_norm**2).sum(axis=0)
x_grid = np.linspace(chisq_approx.min(), chisq_approx.max(), num=200)
chisq_theory = stats.chi2.pdf(x_grid, dof)

fig, ax = plt.subplots()
ax.hist(chisq_approx, np.arange(chisq_approx.min(), chisq_approx.max()+1, 1), align='left',
        density=True, histtype='stepfilled', linewidth=2, color='black', label='Sampled Distribution')
ax.plot(x_grid, chisq_theory, linewidth=4, color='red', label=r'$\chi^2$' + ' Distribution')
ax.set_xlim(50, 160)
ax.set_ylim(0, 0.032)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$pdf(x)$')
ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.002))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob1.eps', dpi=500)