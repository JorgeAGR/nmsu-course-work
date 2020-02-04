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

def fitLine(x, y):
    g = np.ones((len(x),2))
    g[:,1] = x
    d = y.reshape(len(y),1)
    m = np.dot(np.dot(np.linalg.inv(np.dot(g.T, g)), g.T), d)
    
    y_model = np.dot(g, m).flatten()
    r = y - y_model
    var = (r**2).sum() / (len(x) - 2)
    m_var = np.linalg.inv(np.dot(g.T, g)) * var
    return m, m_var

def linear(x, m):
    g = np.ones((len(x),2))
    g[:,1] = x
    d = np.dot(g, m).flatten()
    return d

# 1a
x = np.array([1, 3, 4, 6, 8, 9, 11, 14])
y = np.array([1, 2, 4, 4, 5, 7, 8, 9])

m, m_var = fitLine(x, y)

fig, ax = plt.subplots()
ax.scatter(x, y, color='black', marker='.')
ax.plot(np.linspace(0, 15), linear(np.linspace(0, 15), m), '--', color='red', label=r'Fit for $y$')
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.text(10, 4, 'Red Fit:')
ax.text(10, 3.5, 'a = {:.2}'.format(m[0][0]) + r'$\pm$' + '{:.2}'.format(np.sqrt(m_var[0,0])))
ax.text(10, 3, 'b = {:.2}'.format(m[1][0]) + r'$\pm$' + '{:.2}'.format(np.sqrt(m_var[1,1])))

# 1b
m, m_var = fitLine(y, x)

ax.plot(linear(np.linspace(0, 15), m), np.linspace(0, 15), '--', color='blue', label=r'Fit for $x$')
ax.text(10, 2, 'Blue Fit:')
ax.text(10, 1.5, 'a = {:.2}'.format(m[0][0]) + r'$\pm$' + '{:.2}'.format(np.sqrt(m_var[0,0])))
ax.text(10, 1, 'b = {:.2}'.format(m[1][0]) + r'$\pm$' + '{:.2}'.format(np.sqrt(m_var[1,1])))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob1.eps', dpi=500)

# 2a
np.random.seed(0)
datasets = np.random.random(size=(100,50))
data = datasets.sum(axis=0)

# 2b
mean = data.mean()
std = data.std()
var = std**2

print('Mean:', mean)
print('Std:', std)
print('Var:', var)

fig, ax = plt.subplots()
ax.plot(np.arange(len(data))+1, data, '.', color='black')
ax.axhline(50, color='gray', label='Expected Mean')
ax.axhline(mean, linestyle='--', color='red', label=r'$\mu$')
ax.fill_between(np.arange(len(data)+5), mean+std, mean-std, color='lightpink', label=r'$\sigma$')
ax.text(18, 44, r'$\mu = $' + '{:.4}'.format(mean))
ax.text(18, 43.5, r'$\sigma^2 = $' + '{:.3}'.format(var))
ax.set_xlim(0,51)
ax.set_ylim(43,55)
ax.set_xlabel('Index')
ax.set_ylabel('x')
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob2.eps', dpi=500)