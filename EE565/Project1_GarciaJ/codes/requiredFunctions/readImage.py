#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:59:27 2019

@author: jorgeagr
"""
import numpy as np
import matplotlib.image as mpimg

def readImage(imgdir):
    img = mpimg.imread(imgdir)[:,:,:3]
    
    num_pix = img.shape[0] * img.shape[1]
    channels = [img[:,:,i].flatten().reshape(1, num_pix) for i in range(3)]
    img_pix = np.zeros((channels[0].shape[1], 3))
    for i in range(3):
        img_pix[:,i] = channels[i]
    
    return img, img_pix