#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 04:33:00 2020

@author: jorgeagr
"""

import numpy as np

A = np.array([[1., 2.],
              [2., 1.]])

eigval, P = np.linalg.eigh(A)

L = np.eye(2) * eigval

print('A = {}'.format(A))
print('PLP^T = {}'.format(P@L@P.T))

Ainv = np.linalg.inv(A)
PLPTinv = np.linalg.inv(P@L@P.T)

print('A^-1 = {}'.format(Ainv))
print('(PLP^T)^-1 = {}'.format(PLPTinv))