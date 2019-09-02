# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:44:39 2019

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

# Metal rod at room temperate, heat equation soln

K = 1
T_0 = 0

def heat(x, t):
    
    return np.exp(-x**2 / (4 * K * t)) / np.sqrt(t) + T_0

x = np.linspace(-12, 12, num = 500)
t = [0.5, 1, 10, 100]
styles = ['.', '-.', '--', '-']

fig, ax = plt.subplots()
ax.set_title(r'$T(x, t) = \sqrt{t^{-1}}e^{-x^2 / 4 t}$')
for i, time in enumerate(t):
    ax.plot(x, heat(x, time), styles[i] , label = 't = ' + str(time) + ' s',)
ax.text(0.8, 1.2, 't = 0.5 s')
ax.text(1.8, 0.5, 't = 1 s')
ax.text(4.2, 0.23, 't = 10 s')
ax.text(10, 0.1, 't = 100 s')
ax.set_xlim(-12, 12)
ax.set_xlabel('x')
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax.set_ylim(0, 1.5)
ax.set_ylabel(r'$\Delta T$')
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.legend()
plt.tight_layout()