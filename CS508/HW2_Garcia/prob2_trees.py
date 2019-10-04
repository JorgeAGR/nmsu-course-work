# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:51:14 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

# Load train data. Split into X matrix and Y array
train_data = pd.read_csv('data/glass_train.csv', index_col=0)
train_x = train_data.iloc[:,:-1].values
train_y = train_data.iloc[:,-1].values

# Load test data. Split into X matrix and Y array.
test_data = pd.read_csv('data/glass_test.csv', index_col=0)
test_x = test_data.iloc[:,:-1].values
test_y = test_data.iloc[:,-1].values

# Define array of max depths to use
max_depths = [i for i in range(2,11)]
max_depths.append(15), max_depths.append(20), max_depths.append(25)

# Function to evaluate decision tree performance for different criterions and
# max depth. Trains and evaluates N times to obtain a statistical description
# of model performance and take into account different convergence due to
# random initializations.
def evaluate_depths(criterion, trials=50):
    
    # Initialize accuracy arrays for train and test data.
    # Column 0: Accuracy mean
    # Column 1: Accuracy standard deviation
    train_acc = np.zeros((len(max_depths), 2))
    test_acc = np.zeros((len(max_depths), 2))
    
    # Iterate through max depths
    for i, depth in enumerate(max_depths):
        # Error for the particular depth
        tr_err = np.zeros(trials)
        te_err = np.zeros(trials)
        # Iterate N times to obtain statistical behavior
        for j in range(trials):
            tree_entropy = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
            tree_entropy.fit(train_x, train_y)
            tr_err[j] = tree_entropy.score(train_x, train_y)
            te_err[j] = tree_entropy.score(test_x, test_y)
        # Calculate and save mean and standard deviation of accuracy
        train_acc[i,0] = tr_err.mean()
        train_acc[i,1] = tr_err.std()
        test_acc[i,0] = te_err.mean()
        test_acc[i,1] = te_err.std()
        
    return train_acc, test_acc

# Define criterions to use for the decision trees
criterions = ['entropy', 'gini']

# Evaluate decision tree accuracy as a function of criterion and depth.
# Plot accuracy results
for criterion in criterions:
    train_acc, test_acc = evaluate_depths(criterion)
    
    fig, ax = plt.subplots()
    ax.errorbar(max_depths, train_acc[:,0], yerr=train_acc[:,1], label='Training',
                color='blue', marker='o', markersize=5, markeredgewidth=2)
    ax.errorbar(max_depths, test_acc[:,0], yerr=test_acc[:,1], capsize=5, label='Testing',
                color='red', marker='o', markersize=5, markeredgewidth=2)
    ax.set_xlim(1, 26)
    ax.set_ylim(0.4, 1.02)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
    ax.set_xlabel('Max Tree Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title(criterion)
    ax.legend()
    fig.tight_layout()
    plt.savefig('figs/prob2_' + criterion + '.eps', dpi=500)