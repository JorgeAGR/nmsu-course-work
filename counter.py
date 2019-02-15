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

volts = 600
counts = [13, 4, 8, 3, 13, 16, 8, 10, 9, 8, 11, 11, 15, 6, 5, 11, 7, 8, 9, 10, 
          7, 6, 12, 15, 14, 12, 5, 14, 9, 8, 10, 12, 11, 7, 9, 6, 16, 11, 13, 
          12, 9, 13, 12, 13, 3, 13, 6, 11, 10, 12, 17, 7, 15, 9, 15, 9, 5, 13, 
          6, 9, 12, 15, 13, 11, 15, 12, 12, 9, 7, 8, 12, 7, 8, 11, 10, 14, 6,
          2, 6, 10, 13, 7, 8, 10, 9, 7, 12, 7, 10, 11, 6, 8, 13, 13, 10, 10, 10,
          11, 8, 13] # per 6 sec

background = 1959 # per 50 min

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
    return np.exp(-mu) * mu ** r / np.math.factorial(r)

fig, ax = plt.subplots()
ax.set_title('Count Hist', weight='bold')
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
ax2.set_title('Running Avg', weight='bold')
ax2.errorbar(np.arange(1, len(counts) + 1), running, yerr = running_s)
ax2.set_xlabel('Number of data points')
ax2.set_ylabel('Average')
ax2.set_xlim(0, 105)
ax2.set_ylim(0, 15)
ax2.grid()

N = len(counts)
    
mus = np.arange(0.2, np.max(counts), 0.2)
mu = mus[np.argmax(maxlikelyhood(mus))]
fig3, ax3 = plt.subplots()
ax3.set_title('Max Likelyhood', weight='bold')
ax3.scatter(mus, maxlikelyhood(mus))
ax3.plot(mu, maxlikelyhood(mu), 'ro')
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel('ln(L)')
ax3.set_xlim(0, 17.5)
ax3.set_ylim(-2000, 1500)
ax3.grid()

chisqs = []
for m in mus:
    cs = chisq(m, values, freqs/100, np.sqrt(freqs)/100)
    chisqs.append(cs)
fig4, ax4 = plt.subplots()
ax4.set_title('Chi Sq', weight='bold')
ax4.scatter(mus, chisqs)
ax4.plot(mus[np.argmin(chisqs)], np.min(chisqs), 'ro')
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel(r'$\chi^2$')
ax4.set_xlim(0, 17.5)
ax4.set_ylim(94,101)
ax4.grid()

poisson_dist = []
for r in values:
    poisson_dist.append(poisson(mu, r))
poisson_dist = np.asarray(poisson_dist)
ax.plot(values, poisson_dist * 100, 'r')