# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:49:22 2019

@author: jorge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train_images.csv', header=None)
df_labels = pd.read_csv('train_labels.csv')
df_test = pd.read_csv('test_images.csv', header=None)
df_test_labels = pd.read_csv('test_labels.csv')

volcano_ind = np.where(df_labels['Volcano?'].values == 1)[0]
not_ind = np.where(df_labels['Volcano?'].values == 0)[0]

train_pix = df.values / 255
train_img = train_pix.reshape(len(df), 110, 110, 1)[:,2:,2:]
test_pix = df_test.values / 255
test_img = test_pix.reshape(len(df_test), 110, 110, 1)[:,2:,2:]
train_y = df_labels.values[:,0]
test_y = df_test_labels.values[:,0]

rf = RandomForestClassifier(bootstrap=True)
rf.fit(train_pix, df_labels.values[:,0])
print('train score:', rf.score(train_pix, train_y))
print('test score:', rf.score(test_pix, test_y))