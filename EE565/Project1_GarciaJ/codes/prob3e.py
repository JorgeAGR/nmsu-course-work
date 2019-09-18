#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:13:24 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.kMeans import KMeans
from requiredFunctions.readImage import readImage
from requiredFunctions.remapImage import remapImage
import matplotlib as mpl
import matplotlib.pyplot as plt

height = 10
width = 8

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

img1, img1_pix = readImage('../data/machine-learning-1.png')
img2, img2_pix = readImage('../data/Nature-Brain.png')
img, img_pix = readImage('../data/nature-1.png')

images_ex = [img1, img2]
image_ex_pixels = [img1_pix, img2_pix]

bits =     [ 2,  3,  4, 5, 6, 7, 8]
max_iter = [20, 15, 10, 6, 4, 2, 1]
for n, img_ex in enumerate(images_ex):
    for i, b in enumerate(bits):
        print('Remapping to', b, 'bits')
        kmeans = KMeans()
        _, iters = kmeans.fit_batch(image_ex_pixels[n], 2**b, converge=1e-2, max_iter=max_iter[i], seed=1)
        pix_mapping = kmeans.predict_cluster(img_pix)
        mean_pixels = kmeans.centroids[pix_mapping]
        
        img_new = remapImage(img, kmeans)
        
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(img_new)
        ax[0].set_title(str(b) + ' bits')
        ax[0].axis('off')
        
        pix_change = np.sqrt(((img_new - img)**2).sum(axis=2))
        if i == 0:
            max_change = pix_change.max()
        pix_change = pix_change / max_change
        diff = ax[1].imshow(pix_change, vmin=0, vmax=1)
        cbar = fig.colorbar(diff, orientation='horizontal', fraction=0.047, pad=0.01)
        cbar.set_label(r'$\Delta d_{RGB}$')
        ax[1].axis('off')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        #plt.savefig('../prob3e_' + 'img_' + str(n) + '_' + str(b) + 'bit.eps')