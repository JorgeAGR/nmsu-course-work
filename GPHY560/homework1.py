# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:34:37 2020

@author: jorge
"""

import numpy as np
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

# 2a
np.random.seed(0)
datasets = np.random.random(size=(100,50))
data = datasets.sum(axis=0)

fig, ax = plt.subplots()
ax.plot(np.arange(len(data))+1, data, '.', color='black')
ax.set_xlim(0,51)
ax.set_ylim(43,55)
ax.set_xlabel('Index')
ax.set_ylabel('x')
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))

# 2b
mean = data.mean()
std = data.std()
var = std**2

print('Mean:', mean)
print('Std:', std)
print('Var:', var)