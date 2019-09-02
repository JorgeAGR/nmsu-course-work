# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:34:07 2019

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

def linear(x, c0, c1):
    return c0 + c1 * x

c = np.array([0.01, 0.008, 0.006, 0.004, 0.002])

t = np.array([29, 37, 44, 56, 63])

a = 2 - np.log10(t)

#e = a / b / c

param, error = curve_fit(linear, c, a)

b = 1.27 # cm

epsilon = param[1] / b

print('Linear Model Slope:', param[1])
print('Molar Absorptivity:', epsilon)

x = np.linspace(0, 0.012, num = 500)
fig, ax = plt.subplots()
ax.scatter(c, a, marker='.', color='black')
ax.plot(x, linear(x, param[0], param[1]), '--', color='gray')
ax.set_xlabel('Concentration [M]')
ax.set_ylabel('Absorption')
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.0005))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.set_xlim(0, 0.012)
ax.set_ylim(0.08, 0.6)
plt.tight_layout()