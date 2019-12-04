#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:14:09 2019

@author: jorgeagr
"""

import numpy as np
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
data = np.loadtxt('../data/animals.txt').T
names = ['Dove', 'Hen', 'Duck', 'Goose', 'Owl', 'Hawk', 'Eagle', 'Fox', 'Dog',
         'Wolf', 'Cat', 'Tiger', 'Lion', 'Horse', 'Zebra', 'Cow', 'Bear']
nrows = 13
ncols = 13
'''
NOTE! Library used requires to run setup.py first (located in requiredFunctions)
'''
som = sompy.SOMFactory.build(data, (nrows, ncols))
som.train(verbose=0)
u_matrix = som.build_u_matrix(som)
xx, yy = np.meshgrid(np.arange(0,nrows, 1), np.arange(0,ncols, 1))

# Part A
all_nodes = som.bmu_ind_to_xy(som.find_bmu(data)[0])[:,:2]
fig, ax = plt.subplots()
ax.axis('off')
ax.scatter(xx, yy, u_matrix*1000, edgecolor='gray', marker='h', facecolors='none')
for i in range(len(all_nodes)):
    ax.text(all_nodes[i,0], all_nodes[i,1], names[i], horizontalalignment='center')
fig.tight_layout(pad=0.5)
fig.savefig('../prob1b.eps', dpi=500)

# Part B
# Feautre 1
sizes = np.asarray([som.codebook.matrix[:,0].reshape(nrows, ncols).T,
                    som.codebook.matrix[:,1].reshape(nrows, ncols).T,
                    som.codebook.matrix[:,2].reshape(nrows, ncols).T])
size_labels = sizes.argmax(axis=0)

fig2 = plt.figure()
ax2, ax3, ax4 = [plt.subplot2grid((4,4), (0,0), colspan=2, rowspan = 2, fig=fig2),
                 plt.subplot2grid((4,4), (0,2), colspan=2, rowspan = 2, fig=fig2),
                 plt.subplot2grid((4,4), (2,1), colspan=2, rowspan = 2, fig=fig2)]
ax2.axis('off')
ax2.scatter(xx, yy, u_matrix*250, c=size_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax2.text(9, 5, 'Small', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax2.text(6, 11, 'Medium', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax2.text(2, 5, 'Large', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')

# Feature 2
legs = np.asarray([som.codebook.matrix[:,3].reshape(nrows, ncols).T,
                   som.codebook.matrix[:,4].reshape(nrows, ncols).T])
legs_labels = legs.argmax(axis=0)

ax3.axis('off')
ax3.scatter(xx, yy, u_matrix*250, c=legs_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax3.text(4, 9, '4 Legs', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax3.text(10, 3, '2 Legs', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')

# Feature 3
hunts = np.asarray([som.codebook.matrix[:,-4].reshape(nrows, ncols).T,
                       -som.codebook.matrix[:,-4].reshape(nrows, ncols).T])
hunts_labels = hunts.argmax(axis=0)

ax4.axis('off')
ax4.scatter(xx, yy, u_matrix*250, c=hunts_labels, cmap=cmap, 
            edgecolor='gray', marker='h', facecolors='none')
ax4.text(10, 3, "Don't Hunt", fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
ax4.text(3, 5, 'Hunt', fontsize='medium', fontweight='bold',
         horizontalalignment='center', verticalalignment='center')
fig2.tight_layout(pad=0.5)
fig2.savefig('../prob1c.eps', dpi=500)