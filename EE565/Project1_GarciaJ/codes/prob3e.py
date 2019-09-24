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

img1, img1_pix = readImage('../data/machine-learning-1.png')
img2, img2_pix = readImage('../data/Nature-Brain.png')
img, img_pix = readImage('../data/nature-1.png')

images_ex = [img1, img2]
image_ex_pixels = [img1_pix, img2_pix]

bits =     6
max_iter = 10
fig, ax = plt.subplots(nrows=2)
for n, img_ex in enumerate(images_ex):
    print('Remapping to', bits, 'bits')
    kmeans = KMeans()
    _, iters = kmeans.fit_batch(image_ex_pixels[n], 2**bits, converge=1e-2, max_iter=max_iter, seed=1)
    pix_mapping = kmeans.predict_cluster(img_pix)
    mean_pixels = kmeans.centroids[pix_mapping]
    
    img_new = remapImage(img, kmeans)
    
    ax[n].imshow(img_new)
    ax[n].set_title(str(bits) + ' bits')
    ax[n].axis('off')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig('../prob3e.eps')