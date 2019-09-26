#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:58:49 2019

@author: jorgeagr
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_images.csv', header=None)
df_labels = pd.read_csv('train_labels.csv')

img = df.loc[0,:].values.reshape((110, 110))
