# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:08:15 2019

@author: jorge
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

spikes = pd.read_csv('../data/spikes.csv', header=None)
data = spikes.values

components = [1, 2, 3]
for c in components:
    data_trans = PCA(n_components=c, random_state=0).fit_transform(data)