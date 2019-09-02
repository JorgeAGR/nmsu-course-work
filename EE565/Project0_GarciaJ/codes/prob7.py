#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:12:06 2019

@author: jorgeagr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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

arrays = []
with open('../data/spikes.csv') as file:
    for line in file:
        arrays.append(line)
for i, line in enumerate(arrays):
    arrays[i] = np.array(line.rstrip('\n').split(','), dtype=np.float)
arrays = np.asarray(arrays)

fig, ax = plt.subplots()
for array in arrays:
    ax.plot(array)
ax.set_xlim(0, 25)
ax.set_ylim(-1.2e-4, 1.2e-4)
ax.ticklabel_format(axis='y', style='sci',
                    scilimits=(0,0), useMathText=True)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5e-4))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1e-4))
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
plt.tight_layout()