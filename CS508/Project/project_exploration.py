#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:58:49 2019

@author: jorgeagr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('train_images.csv', header=None)
df_labels = pd.read_csv('train_labels.csv')

dataset_pixels = df.values.flatten()

volcano_ids = np.where(df_labels['Volcano?'].values == 1)[0]

'''
for i in range(10):
    img = df.loc[volcano_ids[i],:].values.reshape((110, 110))
    kmeans = KMeans(n_clusters=2**2)
    kmeans.fit(img.flatten().reshape(-1,1))
    img_new = kmeans.cluster_centers_[kmeans.predict(img.flatten().reshape(-1,1))].flatten().reshape(110, 110)
    fig, ax = plt.subplots()
    ax.imshow(img_new)
    f, a = plt.subplots()
    a.plot((img_new.mean(axis=0)+img_new.mean(axis=1))/2)
    
'''

pca = PCA(n_components=10)
pca.fit(df.values)

df_trans = pca.transform(df.values)

plt.scatter(df[4], df[1], c=df_labels.values[:,0])