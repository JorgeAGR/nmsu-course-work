#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:54:12 2019

@author: jorgeagr
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requiredFunctions.sompy as sompy
# Settings for plots
width = 10
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18
cmap = plt.get_cmap('tab20')
cmap = mpl.colors.ListedColormap(cmap((1, 7, 5)))

# Data in column form, transpose into row form
data = pd.read_csv('../data/starwars.csv')
names = data.loc[:,'species'].values
data = np.asarray(data.values[:,1:], dtype=np.int)
nrows = 18
ncols = 18
'''
NOTE! Library used requires to run setup.py first (located in requiredFunctions)
'''
som = sompy.SOMFactory.build(data, (nrows, ncols))
som.train(verbose=0, train_rough_radiusin=2, train_finetune_radiusin=2,
          train_rough_len=2000, train_finetune_len=0)
u_matrix = som.build_u_matrix(som)
xx, yy = np.meshgrid(np.arange(0,nrows, 1), np.arange(0,ncols, 1))

# Part A
all_nodes = som.bmu_ind_to_xy(som.find_bmu(data)[0])[:,:2]
fig, ax = plt.subplots()
ax.axis('off')
ax.scatter(xx, yy, u_matrix*800, edgecolor='gray', marker='h', facecolors='none')
for i in range(len(all_nodes)):
    ax.text(all_nodes[i,1], all_nodes[i,0], names[i], rotation=-45,
            horizontalalignment='center', verticalalignment='center')
fig.tight_layout(pad=0.5)
fig.savefig('../prob2a.eps', dpi=500)

# Part B
fig2 = plt.figure()

ax2, ax3, ax4 = [plt.subplot2grid((4,4), (0,0), colspan=2, rowspan = 2, fig=fig2),
                 plt.subplot2grid((4,4), (0,2), colspan=2, rowspan = 2, fig=fig2),
                 plt.subplot2grid((4,4), (2,1), colspan=2, rowspan = 2, fig=fig2)]
# Feature 1
species = np.asarray([som.codebook.matrix[:,0].reshape(nrows, ncols),
                      som.codebook.matrix[:,1].reshape(nrows, ncols),
                      som.codebook.matrix[:,2].reshape(nrows, ncols)])
species_labels = species.argmax(axis=0)

ax2.axis('off')
ax2.scatter(xx, yy, u_matrix*175, c=species_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax2.text(3, 15, 'Reptilian', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax2.text(6, 5, 'Mammal', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax2.text(9, 13, 'Amphibian', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')

# Feature 2
fingers = np.asarray([som.codebook.matrix[:,18].reshape(nrows, ncols),
                   -som.codebook.matrix[:,18].reshape(nrows, ncols)])
fingers_labels = fingers.argmax(axis=0)

ax3.axis('off')
ax3.scatter(xx, yy, u_matrix*175, c=fingers_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax3.text(8, 10, '5 Fingers', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax3.text(16, 6, 'Other Fingers', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
#fig3.tight_layout(pad=0.5)

# Feautre 1
sizes = np.asarray([som.codebook.matrix[:,3].reshape(nrows, ncols),
                    som.codebook.matrix[:,4].reshape(nrows, ncols),
                    som.codebook.matrix[:,5].reshape(nrows, ncols)])
size_labels = sizes.argmax(axis=0)

ax4.axis('off')
ax4.scatter(xx, yy, u_matrix*175, c=size_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax4.text(4, 10, 'Medium', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center',)
ax4.text(11, 8, 'Small', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax4.text(16, 6, 'Tall', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')

fig2.tight_layout(pad=0.5)
fig2.savefig('../prob2b.eps', dpi=500)