# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:34:40 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Load data
data_raw = pd.read_csv('data/google_review_ratings.csv')
# New dataframe that excludes User column
data_ratings = data_raw.loc[:, data_raw.keys()[1:]]

# 1a
# Drop NaN values
data_ratings = data_ratings.dropna()

# 1b
# Initialize scaler and transform data to z-score
scaler = StandardScaler()
z_scores = scaler.fit_transform(data_ratings.values)
# Find ooutliers and remove them
outliers = (z_scores > 3).sum(axis=1)
data_ratings = data_ratings.loc[outliers == 0]

# Save new data set
data_ratings.to_csv('data.csv', index=False)