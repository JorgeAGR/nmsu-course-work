#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:22:32 2019

@author: jorgeagr
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 14

# Load data and replace class string with integer
data = pd.read_csv('data/iris.csv', header=None, index_col=None)
data.replace('Iris-setosa', 1, inplace=True)
data.replace('Iris-versicolor', 0, inplace=True)
data.replace('Iris-virginica', 0, inplace=True)

x = data.loc[:, :3].values
y = data.loc[:, 4].values

# Split into 80/20 Train/Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit a Logistic Regression model to the training data
log_reg = LogisticRegression(solver='liblinear', random_state=0)
log_reg.fit(x_train, y_train)

# Predict class probability for the testing set and evaluate the ROC curve
test_prob = log_reg.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, test_prob[:,1])

# Plotting routine for ROC
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='Setosa ROC')
ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='black', label='Chance')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend()
plt.savefig('figs/prob2_logreg_roc.eps', dpi=500)


# Appendix Code: Plot the scatter plot matrix of the data set; I was skpetical
# about the perfect classification but the classes are linearly seperable.
plot_scatter = False
if plot_scatter:
    cmap = plt.get_cmap('tab10')
    plot_colors = mpl.colors.ListedColormap(cmap([0, 3]))
    fig_scat, ax_scat = plt.subplots()
    pd.plotting.scatter_matrix(data.iloc[:,:4], c=data.iloc[:,4],
                               cmap=plot_colors, ax=ax_scat, 
                               hist_kwds={'bins':10, 'histtype':'step',
                                          'color':'black', 'linewidth':2})
    fig_scat.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('figs/prob2_scatter.eps', dpi=500)