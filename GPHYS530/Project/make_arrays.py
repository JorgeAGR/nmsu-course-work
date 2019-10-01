#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:05:08 2019

@author: jorgeagr
"""

import os
import numpy as np
import obspy

directories = ['data/', 'data/train/', 'data/test/']
for d in directories:
    try:
        os.mkdir(d)
    except:
        pass

train_dir = '../../../seismograms/SS_kept/'
test_dir = '../../../seismograms/SS_kept_test/'
train_files = np.sort(os.listdir(train_dir))
test_files = np.sort(os.listdir(test_dir))

resample_Hz = 10
data_points = 5000

def sac2npy(directory, files):
    seismos = np.zeros(len(files), data_points)
    for i, file in enumerate(files):
        seis = obspy.read(train_dir + file)
        seis = seis[0].resample(resample_Hz).detrend()
        if seis.data.shape > data_points:
            seis.data = seis.data[1:]
        seismos[i] = seis.data
    
    return seismos

train_seismos = sac2npy(train_dir, train_files)
test_seismos = sac2npy(test_dir, test_files)