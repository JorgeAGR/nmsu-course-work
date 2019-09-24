#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:50:41 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kMeans import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 10
width = 10

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

N = 50

data1 = circGauss(N//2, 3, 0, 0, seed=1)
data2 = circGauss(N//2, 3, 5, 5, seed=2)
data = np.vstack((data1, data2))
trials = 1000

iterations = np.zeros(trials)

for i in range(trials):
    kmeans = KMeans()
    _, iters = kmeans.fit_batch(data, 2)
    iterations[i] = iters
    
print('Average number of iterations to converge:', iterations.mean())
print('Minimum number of iterations:', iterations.min())
print('Maximum number of iterations:', iterations.max())