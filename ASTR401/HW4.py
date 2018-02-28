#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:12:12 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib.pyplot as plt

def prob2(tau, N):
    I = [1]
    for n in range(1,N):
        i = I[-1] * np.exp(-tau)
        I.append(i)
    return I

tau = np.array([0.1, 1, 10])
N = 1000

I1 = prob2(tau[0], N)
I2 = prob2(tau[1], N)
I3 = prob2(tau[2], N)

plt.plot(range(N),I1)
plt.plot(range(N),I2)
plt.plot(range(N),I3)