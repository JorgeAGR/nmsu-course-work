#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:23:58 2018

@author: jorgeagr
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#from scipy.stats import poisson

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

class LSF(object):
    
    def __init__(self, x, y, sx, sy, fit='poly', n=1):
        
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
        self.fit = fit
        self.n = n # Should be 1 unless dealing with polys
        
        self.fit_poly()
        self.evaluate()
        
        print(self.a, self.da, self.rechisq)
    
    def fit_poly(self):
        # Data Vector (Outputs)
        X = np.zeros(shape = (self.n+1, 1))
        # Measurement Matrix (Inputs)
        M = np.zeros(shape = (self.n+1, self.n+1))
        # Coeffficients Vector
        self.a = np.zeros(self.n+1)
        counter = 0
        while True:
            counter += 1
            #dydx = np.zeros(len(x))
            A = self.a.copy()
            self.w = 1 / (self.sy ** 2 + (self.dfunc(self.x) * self.sx)**2)
            for i in range(self.n+1):
                for j in range(self.n+1):
                    M[i][j] = np.sum(self.w * self.x**(j+i))
                X[i][0] = np.sum(self.w * self.y * self.x ** i)
            self.a = np.dot(np.linalg.inv(M), X).flatten()
            if (np.abs(A - self.a).all() < 1e-12) or (counter == 100):
                break
        
        self.a = self.a.flatten()
        self.da = np.sqrt(np.linalg.inv(M).diagonal())
        
    def evaluate(self):
        self.chisq = np.sum((self.y - self.func(self.x))**2 * self.w) # chisq matters too!
        self.rechisq = self.chisq / (len(self.x) - self.n + 1) # Check this? Apparent DOF is not just on x & y
        
        #WSSR - Weighted sum square residuals
        
    def func(self, x):
        def linear():
            return self.a[0] + self.a[1] * x
        
        def poly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * x ** (i)
            return f
        
        def trig():
            return self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        funcs = {'linear': linear,
                'poly': poly,
                'trig': trig,}
        return funcs.get(self.fit)()
    
    def dfunc(self, x):
        def dlinear():
            return self.a[1]
        
        def dpoly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * i * x ** (i-1)
            return f
        
        def dtrig():
            return -self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        dfuncs = {'linear': dlinear,
                  'poly': dpoly,
                  'trig': dtrig,}
        return dfuncs.get(self.fit)()

def maxlikelyhood(mu):
    lnL = -N * mu + np.log(mu) * np.sum(counts)
    return lnL

def chisq(mu, r, y, std): #for poisson only
    chi = 0
    for i in range(len(r)):
        if std[i] == 0:
            std[i] = 1
        #chi+= (((np.exp(-mu) * mu ** r[i]) / np.math.factorial(r[i]) - y[i]) / std[i])**2
        chi += (poisson(mu, r[i]) - y[i] / std[i])**2
    return chi

def poisson(mu, r):
    return np.exp(-mu) * (mu ** r) / np.math.factorial(r)

volts = 600
counts = [13, 4, 8, 3, 13, 16, 8, 10, 9, 8, 11, 11, 15, 6, 5, 11, 7, 8, 9, 10, 
          7, 6, 12, 15, 14, 12, 5, 14, 9, 8, 10, 12, 11, 7, 9, 6, 16, 11, 13, 
          12, 9, 13, 12, 13, 3, 13, 6, 11, 10, 12, 17, 7, 15, 9, 15, 9, 5, 13, 
          6, 9, 12, 15, 13, 11, 15, 12, 12, 9, 7, 8, 12, 7, 8, 11, 10, 14, 6,
          2, 6, 10, 13, 7, 8, 10, 9, 7, 12, 7, 10, 11, 6, 8, 13, 13, 10, 10, 10,
          11, 8, 13] # per 6 sec

background = 1959 # per 50 min

df = pd.read_csv('potassium.csv', index_col=0)

fig, ax = plt.subplots()
ax.set_title('Data Histogram', weight='bold')
hist = ax.hist(counts, bins = np.arange(np.max(counts)+2),align = 'left',
               color = 'black', histtype = 'step', stacked = True, fill = False,
               linewidth = 2)
values, freqs = hist[1][:-1], hist[0]
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.errorbar(values, freqs, yerr = np.sqrt(freqs), fmt = '.', color = 'black')
ax.set_xlabel('Counts / 6 sec')
ax.set_ylabel('Frequency')
ax.set_xlim(0,np.max(counts) + 1)
ax.set_ylim(0, 16)
ax.grid()
plt.tight_layout()

running = []
running_s = []
for i in range(len(counts)):
    sample = len(counts[0:i+1])
    if sample == 1:
        sample += 1
    avg = np.mean(counts[0:i+1])
    std = np.std(counts[0:i+1], ddof = 1)
    running.append(avg)
    running_s.append(std)
    
fig2, ax2 = plt.subplots()
ax2.set_title('Running Average', weight='bold')
ax2.errorbar(np.arange(1, len(counts) + 1), running, yerr = running_s)
ax2.set_xlabel('Number of data points')
ax2.set_ylabel('Average')
ax2.set_xlim(0, 105)
ax2.set_ylim(0, 15)
ax2.xaxis.set_major_locator(mtick.MultipleLocator(20))
ax2.xaxis.set_minor_locator(mtick.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mtick.MultipleLocator(1))
ax2.grid()
plt.tight_layout()

N = len(counts)

mu = running[-1]
mu_s = running_s[-1]
    
mus = np.arange(0, np.max(counts), 0.5)
#mu = mus[np.argmax(maxlikelyhood(mus))]
fig3, ax3 = plt.subplots()
ax3.set_title('Poisson Distribution: Max Likelihood', weight='bold')
ax3.scatter(mus, maxlikelyhood(mus))
ax3.plot(mus[np.argmax(maxlikelyhood(mus))], np.max(maxlikelyhood(mus)), 'ro')
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel('ln(L)')
ax3.set_xlim(0, 17.5)
ax3.set_ylim(-2000, 1500)
ax3.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax3.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax3.yaxis.set_minor_locator(mtick.MultipleLocator(100))
ax3.grid()
plt.tight_layout()

chisqs = []
for m in mus:
    cs = chisq(m, values, freqs/100, np.sqrt(freqs)/100)
    chisqs.append(cs)
fig4, ax4 = plt.subplots()
ax4.set_title(r'Poisson Distribution: Min $\chi^2$', weight='bold')  
ax4.scatter(mus, chisqs)
ax4.plot(mus[np.argmin(chisqs)], np.min(chisqs), 'ro')
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel(r'$\chi^2$')
ax4.set_xlim(0, 17.5)
ax4.set_ylim(94,101)
ax4.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax4.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
ax4.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
ax4.grid()
plt.tight_layout()

poisson_dist = []
for r in values:
    poisson_dist.append(poisson(mu, r))
poisson_dist = np.asarray(poisson_dist)
ax.plot(values, poisson_dist * 100, 'r', label='Poisson Fit')
ax.legend()

start = 490
end = 720
fitgrid = np.linspace(start, end)
flat = df.loc[start:end, :]
linear = LSF(np.asarray(flat.index.values, dtype=np.float), flat.cpm.values, np.zeros(len(flat)), flat.sig.values)
const = LSF(np.asarray(flat.index.values, dtype=np.float), flat.cpm.values, np.zeros(len(flat)), flat.sig.values, n=0)

fig5, ax5 = plt.subplots()
ax5.set_title('GM High Voltage', weight='bold')
ax5.set_xlabel('Voltage (V)')
ax5.set_ylabel('Counts per minute')
ax5.grid()
ax5.set_ylim(0, 200)
ax5.errorbar(df.index.values, df.cpm, yerr=df.sig, fmt='.', label='Measurements')
ax5.plot(fitgrid, linear.func(fitgrid), label='Linear')
ax5.plot(fitgrid, const.func(fitgrid), label='Average')
ax5.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax5.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax5.yaxis.set_minor_locator(mtick.MultipleLocator(5))
ax5.legend()
plt.tight_layout()