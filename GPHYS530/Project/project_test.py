#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:57:18 2019

@author: jorgeagr
"""

import numpy as np
import obspy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

seismogram = obspy.read('../../../seismograms/SS_kept/19881117A.philippine.HRV.BHT.s_fil')

# Kmeans compression because why not
kmeans = KMeans(n_clusters=2**3)
kmeans.fit(seismogram[0].data.reshape(-1,1))
seis_compressed = kmeans.cluster_centers_[kmeans.predict(seismogram[0].data.reshape(-1,1))]

plt.plot(seis_compressed)
