# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:28:54 2019

@author: jorge
"""

import numpy as np

def oneHotEncode(class_vector):
    
    class_vector = class_vector.astype(np.int)
    
    classes = np.unique(class_vector)
    num_classes = classes.size
    
    class_matrix = np.zeros((len(class_vector), num_classes))
    
    for c in range(num_classes):
        class_matrix[np.where(class_vector == classes[c]), c] = 1
        
    return class_matrix