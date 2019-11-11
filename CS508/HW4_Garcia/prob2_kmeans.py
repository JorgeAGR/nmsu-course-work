# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:31 2019

@author: jorge
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

data = pd.read_csv('data.csv')

