#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:46:44 2020

@author: jorgeagr
"""

import numpy as np
import pandas as pd

house_df = pd.read_csv('kc_house_data.csv')

# Generate indices and shuffle to them to split
indices = np.arange(len(house_df))
np.random.seed(0)
np.random.shuffle(indices)

cutoff = int(np.round(len(house_df)*0.8))
train_df = house_df[:cutoff]
test_df = house_df[cutoff:]

train_df.to_csv('training.csv', index=False)
test_df.to_csv('testing.csv', index=False)
