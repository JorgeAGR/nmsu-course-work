#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:59:42 2019

@author: jorgeagr

Split the dataset into small batches to upload into GitHub; used
as a way to remotely acquire the data when working on Google Colab
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/train_images.csv', header=None)
df_labels = pd.read_csv('data/train_labels.csv')
df_test = pd.read_csv('data/test_images.csv', header=None)
df_test_labels = pd.read_csv('data/test_labels.csv')

df = pd.concat([df, df_test])
df_labels = pd.concat([df_labels, df_test_labels])

for i, batch in enumerate(np.arange(0, 10000, 500)):
    init = batch
    end = batch + 500
    if end > df.shape[0]:
        end = -1
    df_batch = df.iloc[init:end]
    df_labels_batch = df_labels.iloc[init:end]
    df_batch.to_csv('data/images_' + str(i) + '.csv', header=None, index=False)
    df_labels_batch.to_csv('data/labels_' + str(i) + '.csv', header=None, index=False)