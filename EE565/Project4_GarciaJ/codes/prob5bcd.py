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
from mpl_toolkits.mplot3d import Axes3D
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
    pca = PCA(n_components=c, random_state=0)
    data_trans = pca.fit_transform(data)
    print('Variance of data captured: {:.2f}%'.format(pca.explained_variance_ratio_.sum()*100))
    fig, ax = plt.subplots()
    if c == 1:
        ax.plot(data_trans, np.zeros(len(data_trans)), 'k.')
        ax.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position(('data',0))
    elif c == 2:
        ax.plot(data_trans[:,0], data_trans[:,1], 'k.')
        ax.set_ylabel(r'$x_1$')
    else:
        ax = Axes3D(fig)
        ax.scatter(data_trans[:,0], data_trans[:,1], data_trans[:,2], color='black')
        ax.set_ylabel(r'$x_1$')
        ax.set_zlabel(r'$x_2$')
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10
        ax.view_init(elev=24, azim=-26)
    ax.set_xlim(-0.00020, 0.00020)
    ax.set_xlabel(r'$x_0$')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-4,-4))
    if c != 3:
        fig.tight_layout(pad=0.5)
    fig.savefig('../prob5_' + str(c) + 'components.eps', dpi=500)