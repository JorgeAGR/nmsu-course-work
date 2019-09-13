#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:40:15 2019

@author: jorgeagr
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

with open('prob1_output.txt', 'w+') as file:

    data = pd.read_csv('dataR2.csv')
    data = data.round(3)
    data_nc = data[data.keys()[:-1]]
    data_desc = data_nc.describe().round(3)
    print(data_desc.to_string(), end='\n\n', file=file)
        
    data_cov = data_nc.cov().round(3)
    print(data_cov.to_string(), end='\n\n', file=file)
    
    data_corr = data_nc.corr().round(3)
    print(data_corr.to_string(), end='\n\n', file=file)
    
    data.hist(bins=10)
    plt.savefig('histograms.eps', dpi=500)
    
    data_subset = data[['Age', 'BMI', 'Glucose', 'Insulin']]
    
    pd.plotting.scatter_matrix(data_subset, c=data['Classification'])
    plt.savefig('scatter_matrix.eps', dpi=500)
    
    fig, ax = plt.subplots()
    ax = pd.plotting.parallel_coordinates(data, 'Classification', color=('red', 'blue'),
                                          cols=['MCP.1', 'Adiponectin', 'Insulin', 'HOMA', 
                                                'BMI', 'Age', 'Leptin', 'Glucose', 'Resistin'])
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('parallel_coordinates.eps', dpi=500)
    
    
    data_ind = np.arange(0, len(data), 1)
    np.random.shuffle(data_ind)
    rand_sample = data.iloc[data_ind[:10]]
    
    print(rand_sample.to_string(), file=file)
    
    pca = PCA(n_components=2)
    pca.fit(data_nc.values)
    
    data_trans = pca.transform(data_nc.values)
    
with open('latex_tables.txt', 'w+') as file2:
    data_latex = data[data.keys()[:-1]]
    print(data_desc.to_latex(), end='\n\n', file=file2)
    print(data_cov.to_latex(), end='\n\n', file=file2)
    print(data_corr.to_latex(), end='\n\n', file=file2)
    print(rand_sample.to_latex(), end='\n\n', file=file2)