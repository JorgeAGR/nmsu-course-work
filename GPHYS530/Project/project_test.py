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
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape, Conv1DTranspose
from keras.models import Model


def rossNet_CAE(input_length, compression_size):
    input_seis = Input(shape=(input_length, 1))

    conv1_1 = Conv1D(32, kernel_size=21, strides=1,
                     activation='relu')(input_seis)
    conv1_2 = Conv1D(32, kernel_size=21, strides=1,
                     activation='relu')(conv1_1)
    bn1 = BatchNormalization()(conv1_2)
    max1 = MaxPooling1D(pool_size=2)(bn1)

    conv2_1 = Conv1D(64, kernel_size=15, strides=1,
                     activation='relu')(max1)
    conv2_2 = Conv1D(64, kernel_size=15, strides=1,
                     activation='relu')(conv2_1)
    bn2 = BatchNormalization()(conv2_2)
    max2 = MaxPooling1D(pool_size=2)(bn2)

    conv3_1 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu')(max2)
    conv3_2 = Conv1D(128, kernel_size=11, strides=1,
                     activation='relu')(conv3_1)
    bn3 = BatchNormalization()(conv3_2)
    max3 = MaxPooling1D(pool_size=2)(bn3)

    flattened = Flatten()(max3)
    
    encoding = Dense(compression_size, activation='relu')(flattened)
    
    reshaped = Reshape(max3.output_shape)(encoding)

    up1 = UpSampling1D(size=2)(reshaped)
    bn_up1 = BatchNormalization()(up1)
    convup1_1 = Conv1DTranspose(128, kernel_size=11, strides=1,
                     activation='relu')(bn_up1)
    convup1_2 = Conv1DTranspose(128, kernel_size=11, strides=1,
                     activation='relu')(convup1_1)

    up2 = UpSampling1D(size=2)(convup1_2)
    bn_up2 = BatchNormalization()(up2)
    convup2_1 = Conv1DTranspose(64, kernel_size=15, strides=1,
                     activation='relu')(bn_up2)
    convup2_2 = Conv1DTranspose(64, kernel_size=15, strides=1,
                     activation='relu')(convup2_1)

    up3 = UpSampling1D(size=2)(convup2_2)
    bn_up3 = BatchNormalization()(up3)
    convup3_1 = Conv1DTranspose(32, kernel_size=21, strides=1,
                     activation='relu')(bn_up3)
    convup3_2 = Conv1DTranspose(32, kernel_size=21, strides=1,
                     activation='relu')(convup3_1)
    output = Conv1D(1, kernel_size=21, strides=1,
                     activation='rely')(convup3_2)

    model = Model(inputs=input_spectra, outputs=[col_dense[-1], temp[-1],
                                                 density[-1], ionized[-1], metal[-1]])

    model.compile(loss='mean_squared_error',#huber_loss,
                  optimizer=Adam(1e-5/2))

    return model)
