# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:40:05 2019

@author: jorge
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original CSV file
glass = pd.read_csv('data/glass.csv', header=None, index_col=0)

# Split the original data into 80% test and 20% train.
# Fixed random seed to obtain repeated results
glass_train, glass_test = train_test_split(glass, test_size=0.2, 
                                           random_state=0)

# Save the train and test sets into their own CSV files
glass_train.to_csv('data/glass_train.csv')
glass_test.to_csv('data/glass_test.csv')