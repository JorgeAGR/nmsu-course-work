#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:34:38 2020

@author: jorgeagr
"""

import numpy as np
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

# LSF function to calcualte model parameters and covariance matrix
def LSF(x, y, N):
    '''
    Inputs:
    x : array
        input data
    y : array
        output data
    N : int
        order of fit
        
    Outputs:
    m : (N+1)-array
        model weights
    m_var : (N+1)*(N+1) matrix
        covariance matrix
    '''
    G = np.ones((len(x), N+1))
    for i in np.arange(N)+1:
        G[:,i] = x**i
    d = y.reshape(len(y),1)
    m = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), d)
    
    d_model = np.dot(G, m)
    r = d - d_model
    var = (r**2).sum() / (len(x) - N+1)
    m_cov = np.linalg.inv(np.dot(G.T, G)) * var
    return m, m_cov

# Calculate for a model given input data and parameters
def func(x, m):
    N = len(m)
    G = np.ones((len(x),N))
    for i in np.arange(N):
        G[:,i] = x**i
    d = np.dot(G, m).flatten()
    return d

x = np.array([0, 3, 5, 7, 9])
y = np.array([1, 5, 7, 6, 5])

m, m_cov = LSF(x, y, 2)

# Part A
print('Fit Parameters:', m.flatten())

x_grid = np.linspace(0, 10)
fig, ax = plt.subplots()
ax.scatter(x, y, 50, marker='o', color='black')
ax.plot(x_grid, func(x_grid, m), color='red', label='Parabola Fit')
ax.set_xlim(-1, 10)
ax.set_ylim(0, 8)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob6.eps', dpi=500)

# Part B
r = y - func(x, m)
data_var = (r**2).sum() / (len(x) - len(m))
print('Data Variance:', data_var)

# Part C
print('Model Parameters Covariance Matrix:\n', m_cov)