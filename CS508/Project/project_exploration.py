#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:58:49 2019

@author: jorgeagr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

train_images = pd.read_csv('train_images.csv', header=None)
train_labels = pd.read_csv('train_labels.csv')

test_images = pd.read_csv('test_images.csv', header=None)
test_labels = pd.read_csv('test_labels.csv')

dataset_pixels = df.values.flatten()

volcano_ind = np.where(train_labels['Volcano?'].values == 1)[0]
not_ind = np.where(train_labels['Volcano?'].values == 0)[0]

np.random.shuffle(volcano_ind)
np.random.shuffle(not_ind)



'''
for i in range(10):
    img = df.loc[volcano_ind[i],:].values.reshape((110, 110))
    kmeans = KMeans(n_clusters=2**2)
    kmeans.fit(img.flatten().reshape(-1,1))
    img_new = kmeans.cluster_centers_[kmeans.predict(img.flatten().reshape(-1,1))].flatten().reshape(110, 110)
    fig, ax = plt.subplots()
    ax.imshow(img_new)
    f, a = plt.subplots()
    a.plot((img_new.mean(axis=0)+img_new.mean(axis=1))/2)
    
'''

#pca = PCA(n_components=10)
#pca.fit(df.values)
'''
pca = KernelPCA(kernel='rbf', gamma=10)
pca.fit(df.values)

df_trans = pca.transform(df.values)

plt.scatter(df[4], df[1], c=df_labels.values[:,0])
'''
'''
bits = np.arange(0, 255, 1)
for i in range(5):
    img_vol = df.loc[volcano_ind[i],:].values
    img_not = df.loc[not_ind[i],:].values
    
    fig, ax = plt.subplots(nrows=2)
    ax[0].hist(img_vol, bits)
    ax[1].hist(img_not, bits)
'''