#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:07:39 2019

@author: jorgeagr
"""
import numpy as np

def remapImage(image, kmeans_model):
    img_new = np.ones_like(image)
    for i in range(image.shape[0]):
        img_new[i,:,:] = kmeans_model.centroids[kmeans_model.predict_cluster(image[i, :, :])]
    
    return img_new