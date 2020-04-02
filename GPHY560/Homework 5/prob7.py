#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 05:53:59 2020

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Using damped LSF as suggested by Menke (Ch 12)
class LSF(object):
    
    def __init__(self, G, d):
        self.N = G.shape[1]
        self.G = G
        self.d = d
        return
    
    # LSF function to calcualte model parameters and covariance matrix
    def fit(self):
        '''    
        Outputs:
        m : (N+1)-array
            model weights
        m_var : (N+1)*(N+1) matrix
            covariance matrix
        '''
        G = self.G
        GTGinv = np.linalg.inv(G.T @ G)
        
        self.m = (GTGinv @ G.T) @ self.d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        self.dof = (G.shape[0] - self.N)
        self.data_var = (r**2).sum() / self.dof
        
        self.m_cov = GTGinv @ G.T @ G @ GTGinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

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

f_points = 7
filter_points = 4

# Build function
f = np.zeros(f_points)
f[0:4] = np.array([2, 1, 0, -1])

# Build G of function
G = np.zeros((f_points, filter_points))
for i in range(filter_points):
    G[i:i+f_points, i] = f[:f_points-i]
    
# Delta function
d = np.zeros(f_points)
d[0] = 1

# m is the inverse filter
model = LSF(G, d)
m, cov = model.fit()

fig = plt.figure()
ax = [plt.subplot2grid((4,4), (0,0), colspan=2, rowspan=2, fig=fig),
      plt.subplot2grid((4,4), (0,2), colspan=2, rowspan=2, fig=fig),
      plt.subplot2grid((4,4), (2,1), colspan=2, rowspan=2, fig=fig)]
ax[0].plot(f, label=r'$f(t)$')
ax[1].plot(m, label=r'$f(t)^{-1}$')
ax[2].plot(np.convolve(f, m), label='Convolution')
ax[2].plot(d, linestyle=':', label='Delta Pulse')

for i in (0, 2):
    ax[i].xaxis.set_major_locator(mtick.MultipleLocator(2))
    ax[i].xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax[0].yaxis.set_major_locator(mtick.MultipleLocator(1))
ax[0].yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax[2].yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax[2].yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
ax[1].xaxis.set_major_locator(mtick.MultipleLocator(1))
ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax[1].yaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
for a in ax:
    a.set_xlabel(r'$t$')
    a.set_ylabel(r'$A$')
    a.legend()
fig.tight_layout(pad=0.5)
fig.savefig('prob7.eps', dpi=200)

# Part C
print('Data Variance: \nvar = {}'.format(model.data_var))
print('Covariance: \ncov(m) = {}'.format(cov))

# Part D
s = np.array([2, 1, 0, -1, 4, 0, -1, -2, 1, 6, 3, -2, -4, 2, 2, 0, -1, 0, 2, 1, 0, -1, 0, 0, 0])
fig2, ax2 = plt.subplots()
ax2.plot(np.convolve(m, s), label=r'$f^{-1}$*Signal')
ax2.xaxis.set_major_locator(mtick.MultipleLocator(5))
ax2.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax2.yaxis.set_major_locator(mtick.MultipleLocator(1))
ax2.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$A$')
ax2.legend()
fig2.tight_layout(pad=0.5)
fig2.savefig('prob7d.eps', dpi=200)