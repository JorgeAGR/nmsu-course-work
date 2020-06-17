#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:24:58 2020

@author: jorgeagr
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.labelsize'] = 18

imgdir = './ellington.png'
img = mpimg.imread(imgdir)

u, s, vh = np.linalg.svd(img)

fig = plt.figure()
ax = [[plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=1, fig=fig),],
       [plt.subplot2grid((3,3), (1,0), colspan=1, rowspan=1, fig=fig),
        plt.subplot2grid((3,3), (1,1), colspan=1, rowspan=1, fig=fig),
        plt.subplot2grid((3,3), (1,2), colspan=1, rowspan=1, fig=fig)],
        [plt.subplot2grid((3,3), (2,0), colspan=1, rowspan=1, fig=fig),
         plt.subplot2grid((3,3), (2,1), colspan=1, rowspan=1, fig=fig),
         plt.subplot2grid((3,3), (2,2), colspan=1, rowspan=1, fig=fig)]]

ax[1][0].imshow(img, cmap='gray')
ax[0][0].plot(np.arange(len(s)), s)
ax[0][0].set_xlabel('Number')
ax[0][0].set_ylabel('Singular Value')
for i, elements in enumerate([int(round(len(s)*.9)), len(s)//2]):
    ax[1][i+1].imshow(u[:, :elements] @ np.diag(s[:elements]) @ vh[:elements,:], cmap='gray')
ax[2][0].imshow(u[:, elements:len(s)] @ np.diag(s[elements:]) @ vh[elements:,:], cmap='gray')
ax[2][1].imshow(u[:, 0].reshape((img.shape[0], 1)) @ (np.diag(s[0].reshape(1,1)) @ vh[0, :].reshape((1,img.shape[1]))).reshape(1, img.shape[1]), cmap='gray')
ax[2][2].imshow(u[:, -1].reshape((img.shape[0], 1)) @ (np.diag(s[-1].reshape(1,1)) @ vh[-1, :].reshape((1,img.shape[1]))).reshape(1, img.shape[1]), cmap='gray')
#ax.set_title(str(b) + ' bits')
titles = [['Original', '90%', 'Top 50%'],
          ['Bottom 50%', 'Largest SV', 'Smallest SV']]
for i in range(2):
    for j in range(3):
        ax[i+1][j].axis('off')
        ax[i+1][j].set_title(titles[i][j])
#ax[0][0].axis('on')
fig.tight_layout(pad=0.5)
fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig('prob2.pdf')