#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:51:55 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.kMeans import KMeans
from requiredFunctions.readImage import readImage
from requiredFunctions.remapImage import remapImage
import matplotlib as mpl
import matplotlib.pyplot as plt

height = 10
width = 6

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

img, img_pix = readImage('../data/nature-1.png')

bits =     [ 2,  3, 7]
max_iter = [20, 15, 4]
fig, ax = plt.subplots(nrows=3)
for i, b in enumerate(bits):
    print('Remapping to', b, 'bits')
    kmeans = KMeans()
    _, iters = kmeans.fit_batch(img_pix, 2**b, converge=1e-2, max_iter=max_iter[i], seed=1)
    pix_mapping = kmeans.predict_cluster(img_pix)
    mean_pixels = kmeans.centroids[pix_mapping]
    
    img_new = remapImage(img, kmeans)
    
    ax[i].imshow(img_new)
    ax[i].set_title(str(b) + ' bits')
    ax[i].axis('off')
    
fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig('../prob3c.eps')