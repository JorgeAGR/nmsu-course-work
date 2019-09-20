#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:40:15 2019

@author: jorgeagr

Assignment 1 for CS 508
Jorge Garcia
Kelly Sanderson

"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

golden_ratio = (np.sqrt(5) + 1) / 2
width = 15
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
    # Load data onto pandas
    data = pd.read_csv('data/dataR2.csv')
    data_nc = data[data.keys()[:-1]] # Subset of data without 'Classification'
    # Calculate the avg, std, min, max and percentiles of data
    data_desc = data_nc.describe().round(3)
    print('Problem 1a: Data Satistics (Avg, Std, Min & Max)', file=file)
    print(data_desc.to_string(), end='\n\n', file=file)
    
    # Calculate the covariance pairs between data attributes
    data_cov = data_nc.cov().round(3)
    print('Problem 1b: Covariance and correlation pairs', file=file)
    print('Covariance pairs table', file=file)
    print(data_cov.to_string(), end='\n\n', file=file)
    
    # Calculate the correlation pairs between data attributes
    data_corr = data_nc.corr().round(3)
    print('Correlation pairs table', file=file)
    print(data_corr.to_string(), end='\n\n', file=file)
    
    # Plot data histograms
    fig0, ax0 = plt.subplots()
    ax_hist = data_nc.hist(bins=10, ax=ax0, histtype='step',
        color='black', linewidth=2)
    fig0.tight_layout()
    fig0.subplots_adjust(wspace=0.15, hspace=0.4)
    for ax in ax_hist.flatten():
        ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
        ax.grid(False)
    #plt.savefig('figs/histograms.eps', dpi=500)
    print('Problem 1c: Histograms (see figs)', end='\n\n', file=file)
    
    # Generate boxplots of every attribute
    fig1, ax1 = plt.subplots(nrows=3, ncols=3)
    for i, key in enumerate(data_nc.keys()):
        ax1[i//3][i%3].boxplot(data[key])
        ax1[i//3][i%3].set_xlabel(key)
        ax1[i//3][i%3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    #plt.savefig('figs/boxplots.eps', dpi=500)
    print('Problem 1d: Boxplots', end='\n\n', file=file)
    
    # Subset of attributes to make scatter plots of
    data_subset = data[['Age', 'BMI', 'Glucose', 'Insulin']]
    
    # Plot scatter plot pairs
    cmap = plt.get_cmap('Set1')
    plot_colors = mpl.colors.ListedColormap(cmap(np.arange(0, 2, 1)))
    fig2, ax2 = plt.subplots()
    pd.plotting.scatter_matrix(data_subset, c=data['Classification'],
                               cmap=plot_colors, ax=ax2, 
                               hist_kwds={'bins':10, 'histtype':'step',
                                          'color':'black', 'linewidth':2})
    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0, hspace=0)
    #plt.savefig('figs/scatter_matrix.eps', dpi=500)
    print('Problem 1e: Scatter Plot Matrix (see figs)', end='\n\n', file=file)
    
    # Plot parallel coordinates of data
    fig3, ax3 = plt.subplots()
    pd.plotting.parallel_coordinates(data, 'Classification', ax=ax3, color=('red', 'blue'),
                                          cols=['MCP.1', 'Adiponectin', 'Insulin', 'HOMA', 
                                                'BMI', 'Age', 'Leptin', 'Glucose', 'Resistin'])
    ax3.set_xticklabels(ax3.xaxis.get_majorticklabels(), rotation=45)
    ax3.grid(False)
    fig3.tight_layout()
    #plt.savefig('figs/parallel_coordinates.eps', dpi=500)
    print('Problem 1f: Parallel Coordinates Plot (see figs)', end='\n\n', file=file)
    
    # Generate an array of indices, shuffle them and then use the first
    # 10 elements as the indices of the randomly sampled data points
    data_ind = np.arange(0, len(data), 1)
    np.random.shuffle(data_ind)
    rand_sample = data.iloc[data_ind[:10]].round(3)
    print('Problem 2a: Random sample of 10 data points', file=file)
    print(rand_sample.to_string(), end='\n\n', file=file)
    
    # Perform PCA on the data and makes a scatter plot with the new 2 dimensions
    pca = PCA(n_components=2)
    pca.fit(data_nc.values)
    data_trans = pca.transform(data_nc.values)
    fig4, ax4 = plt.subplots()
    pca_scat = ax4.scatter(data_trans[:,0], data_trans[:,1],
                           c=data['Classification'], cmap=plot_colors)
    ax4.set_xlabel(r'$x_1$')
    ax4.set_ylabel(r'$x_2$')
    legend_elements = [mpl.lines.Line2D([0], [0], marker='.', color='white',
                                        markerfacecolor=plot_colors.colors[0], markersize=14, label='Healthy'),
                       mpl.lines.Line2D([0], [0], marker='.', color='white',
                                        markerfacecolor=plot_colors.colors[1], markersize=14, label='Patient')]
    ax4.legend(handles=legend_elements)
    fig4.tight_layout()
    #plt.savefig('figs/pca_scatter.eps', dpi=500)
    print('Problem 2b: PCA Scatter Plot (see figs)', end='\n\n', file=file)

# Writes out table TeX code for report
with open('etc/latex_tables.txt', 'w+') as file2:
    data_latex = data[data.keys()[:-1]]
    print(data_desc.to_latex(), end='\n\n', file=file2)
    print(data_cov.to_latex(), end='\n\n', file=file2)
    print(data_corr.to_latex(), end='\n\n', file=file2)
    print(rand_sample.to_latex(), end='\n\n', file=file2)