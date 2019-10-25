# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 01:05:32 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.train_Perceptron import PerceptronClassifier
from requiredFunctions.doubleMoon import doubleMoon
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

N = 500
r = 1
w = 0.6
d_range  = [0.5, 0, -0.5]

d_acc = np.zeros((3, N))
for n, d in enumerate(d_range):
    trials = 30
    trial_acc = np.zeros((trials, N))
    for i in range(trials):
        data = doubleMoon(N, w, r, d, seed=i)
        x_train, y_train = data[:,:2], data[:,2]
        
        init_weights = np.random.rand(x_train.shape[1] + 1) - 0.5
        perceptron = PerceptronClassifier()
        perceptron.fit_Online(x_train, y_train, w0=init_weights, max_epochs=1)
        trial_acc[i] = perceptron.accuracy_log
    
    # Find average iteration error and plot
    acc_avg = trial_acc.mean(axis=0)
    d_acc[n] = acc_avg
    
iter_grid = np.arange(1, N+1, 1)
fig, ax = plt.subplots()

for n in range(3):
    ax.plot(iter_grid, d_acc[n], label=r'$d =$' + str(d_range[n]))
ax.set_xlim(-20, 520)
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig('../prob2c.eps', dpi=500)