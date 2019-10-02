# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:51:14 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/glass_train.csv', index_col=0)
train_x = train_data.iloc[:,:9].values
train_y = train_data.iloc[:,-1].values

test_data = pd.read_csv('data/glass_test.csv', index_col=0)
test_x = test_data.iloc[:,:9].values
test_y = test_data.iloc[:,-1].values

max_depths = [i for i in range(2,11)]
max_depths.append(15), max_depths.append(20), max_depths.append(25)

def evaluate_depths(criterion, trials=50):
    
    train_acc = np.zeros((len(max_depths), 2))
    test_acc = np.zeros((len(max_depths), 2))
    
    for i, depth in enumerate(max_depths):
        tr_err = np.zeros(trials)
        te_err = np.zeros(trials)
        for j in range(trials):
            tree_entropy = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
            tree_entropy.fit(train_x, train_y)
            tr_err[j] = tree_entropy.score(train_x, train_y)
            te_err[j] = tree_entropy.score(test_x, test_y)
        train_acc[i,0] = tr_err.mean()
        train_acc[i,1] = tr_err.std()
        test_acc[i,0] = te_err.mean()
        test_acc[i,1] = te_err.std()
        
    return train_acc, test_acc

criterions = ['entropy', 'gini']

for criterion in criterions:
    train_acc, test_acc = evaluate_depths(criterion)
    
    fig, ax = plt.subplots()
    ax.errorbar(max_depths, train_acc[:,0], yerr=train_acc[:,1])
    ax.errorbar(max_depths, test_acc[:,0], yerr=test_acc[:,1])

'''
# Part A
entropy_train_acc, entropy_test_acc = evaluate_depths('entropy')

fig, ax = plt.subplots()
ax.errorbar(max_depths, entropy_train_acc[:,0], yerr=entropy_train_acc[:,1])
ax.errorbar(max_depths, entropy_test_acc[:,0], yerr=entropy_test_acc[:,1])


# Part B
gini_train_acc, gini_test_acc = evaluate_depths('gini')

fig2, ax2 = plt.subplots()
ax2.errorbar(max_depths, gini_train_acc[:,0], yerr=gini_train_acc[:,1])
ax2.errorbar(max_depths, gini_test_acc[:,0], yerr=gini_test_acc[:,1])
'''