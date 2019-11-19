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

pca = PCA(n_components=3, svd_solver='full')
data_trans = pca.fit_transform(data)
for c in range(3):
    print('Variance of data captured in {} components: {:.2f}%'.format(c+1,pca.explained_variance_ratio_[:c+1].sum()*100))
    
fig = plt.figure()
ax = [plt.subplot2grid((3,1), (0,0), colspan=1, rowspan = 1, fig=fig),
      plt.subplot2grid((3,1), (1,0), colspan=1, rowspan = 2, fig=fig),]
ax[0].scatter(data_trans[:,0], np.zeros(len(data_trans)), color='black')
ax[0].get_yaxis().set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_position(('data',0))
ax[0].xaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax[0].xaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))

ax[1].scatter(data_trans[:,0], data_trans[:,1], color='black')
ax[1].set_ylim(-0.00020, 0.00020)
ax[1].set_ylabel(r'$x_1$')
ax[1].xaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))
ax[1].yaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))
for a in ax:
    a.set_xlim(-0.00020, 0.00020)
    a.set_xlabel(r'$x_0$')
    a.ticklabel_format(axis='both', style='sci', scilimits=(-4,-4))
fig.tight_layout(pad=0.5)
fig.savefig('../prob5bc.eps', dpi=500)

fig, ax = plt.subplots()
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
ax.xaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))
ax.yaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))
ax.zaxis.set_major_locator(mtick.MultipleLocator(1e-4))
ax.zaxis.set_minor_locator(mtick.MultipleLocator(0.5e-4))
fig.savefig('../prob5d.pdf', dpi=500)