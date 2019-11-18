#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:17:27 2019

@author: jorgeagr
"""
import numpy as np
from requiredFunctions.trainMLP import trainMLP
from requiredFunctions.MLP import MLP
from requiredFunctions.gaussX import gaussX
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

# Data parameters
N_train = 300
N_val = 3000
var = 1

# Model parameters
learning_rate = 1e-3
alpha = 0
epochs = 1000*5

# Init train data
train_data = gaussX(N_train, var, seed=0)
x_train, y_train = train_data[:,:2], train_data[:,2]
x_train, y_train = x_train.T, y_train.reshape(1, len(y_train))

# Init validation data
val_data = gaussX(N_val, var, seed=100)
x_val, y_val = val_data[:,:2], val_data[:,2]
x_val, y_val = x_val.T, y_val.reshape(1, len(y_val))

wh, wo, mse, mse_val = trainMLP(x_train, y_train, [5], learning_rate, alpha,
                                epochs, verbose=False, X_val=x_val, D_val=y_val)

epoch_grid = np.arange(0, epochs) + 1
print('Early Stopping Point at', epoch_grid[np.argmin(mse_val)], 'epochs')
fig, ax = plt.subplots()
ax.plot(epoch_grid, np.log10(mse.flatten()), color='blue', markeredgewidth=2, label='Training')
ax.plot(epoch_grid, np.log10(mse_val.flatten()), '--', color='crimson', markeredgewidth=2, label='Testing')
ax.set_xlabel('Epochs')
ax.set_ylabel(r'$\log_{10}$MSE')
ax.xaxis.set_major_locator(mtick.MultipleLocator(1000))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(250))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.ticklabel_format(axis='x', style='sci', scilimits=(3,3), useMathText=True)
ax.set_xlim(-250,epochs+250)
ax.set_ylim(-0.7,0.6)
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('../prob3b.eps', dpi=500)