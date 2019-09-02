# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:32:13 2019

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
import sklearn.linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

class regression_model(object):
    
    def __init__(self):
        None
        
    def load_csv(self, path):
        # easiest is data is in format of Y, X1, X2...
        self.data = pd.read_csv(path)
        
    def load_arrays(self, **kwarrays):
        self.data = pd.DataFrame()
        for key in kwarrays:
            self.data[key] = kwarrays[key]
        
    def set_vars(self, y_key, *x_vars):
        self.y = self.data.loc[:, y_key]
        self.X = self.data.loc[:, [*x_vars]]
        
    def fit_model(self, intercept=True):
        if intercept:
            cols = np.array(['Intercept'])
            cols = np.append(cols, self.X.columns.values)
            self.X.loc[:, 'Intercept'] = np.ones(len(self.X))
            self.X = self.X[cols]
        self.model_OLS = sm.OLS(self.y, self.X)
        self.fit = self.model_OLS.fit()
        #print(self.fit.summary())
        self.results()
        self.params = self.fit.params.values
        
    def results(self):
        r_sq = self.fit.rsquared
        mult_r = np.sqrt(r_sq)
        std_err = np.sqrt(self.fit.ssr / self.fit.df_resid)
        
        print('== Summary Output ==')
        
        print('\n- Dependent Variable:', self.y.name)
        
        print('\n=====================+======')
        print('- Regression Statistics -')
        print('----------------------------')
        print('Multiple R:        {:.4f}'.format(mult_r))
        print('R Square:          {:.4f}'.format(r_sq))
        print('Adjusted R Square: {:.4f}'.format(self.fit.rsquared_adj))
        print('Standard Error:    {:.4f}'.format(std_err))
        print('Observations:      {}'.format(len(self.y)))
        print('=========================+==')
        
        print('\n===================================================================')
        print('| - ANOVA - |  {:^4} | {:^15} | {:^15} | {:^7} |'.format('df', 'SS', 'MS', 'F'))
        print('|-----------|-------|-----------------|-----------------|---------|')
        print('| Regression| {:5d} | {:15.4f} | {:15.4f} | {:7.3f} |'.format(int(self.fit.df_model), self.fit.ess, 
                                                              self.fit.mse_model, self.fit.fvalue))
        print('| Residual  | {:5d} | {:15.4f} | {:15.4f} | {:7} |'.format(int(self.fit.df_resid), self.fit.ssr, 
                                                               self.fit.mse_resid, ''))
        print('| Total     | {:5d} | {:15.4f} | {:15} | {:7} |'.format(int(self.fit.df_resid + self.fit.df_model), 
                                                              self.fit.ess + self.fit.ssr, '', ''))
        print('===================================================================')
        
        print('\n=====================================================================')
        print('|  - Model -  | {:^12} | {:^12} | {:^9} | {:^9} |'.format('Coefficients', 'Std Error', 't-Stat', 'P-value'))
        print('|-------------|--------------|--------------|-----------|-----------|')
        for i in range(len(self.fit.params.values)):
            print('| {:12}| {:12.4f} | {:12.4f} | {:9.4f} | {:9.7f} |'.format(self.X.keys().values[i],
                                             self.fit.params.values[i],
                                             self.fit.bse.values[i],
                                             self.fit.tvalues.values[i],
                                             self.fit.pvalues.values[i]))
        print('=====================================================================')
    
    def get_params(self):
        return self.fit.params.values
    
    #def func(self, input_matrix): # Not functional
        # input_matrix should be a matrix of which the columns
        # are the various inputs for a coefficient
        #N = x.shape[0]
        #param = x.shape[1]
    #    return np.dot(input_matrix, self.fit.params.values)
    
    #def plot(self, input_key, **kwargs): # Not functional
        # 2-D plot, holding other variables constants'
        # kwargs is of the form:
        # var_key=var_matrix
    #    fig, ax = plt.subplots()
    #    ax.scatter(self.data[input_key], self.y)
    #    for key in kwargs:
    #        x_ind = np.where(self.data.keys() == key)[0][0]
    #        y = self.func(kwargs[key])
    #        ax.plot(kwargs[key][:, x_ind], y)
    
    #def to_latex(self): # Under development
    #    input_lines = [6, 8, 10, 12, 14, 24, 26, 28, 38]
    #    for i, c in self.get_params()[:-1]:
    #        input_lines.append(38 + 2 * (i + 1))
    #    with open('table_template.txt') as file:
    #        lines = file.readlines()
    #        
    #    mult_r = lines[6].split('&')
    #    mult_r

# eg
# data from handout 6, pg. 6
'''
# Y-var
FRIG = np.array([1317, 1615, 1662, 1295, 1271, 1555, 1639, 1238, 1277, 1258,
                 1417, 1185, 1196, 1410, 1417, 919, 943, 1175, 1269, 973, 1102,
                 1344, 1641, 1225, 1429, 1699, 1749, 1117, 1242, 1684, 1764, 1328])

#data
DUR = np.array([252.6, 272.4, 270.9, 273.9, 268.9, 262.9, 270.9,
                263.4, 260.6, 231.9, 242.7, 248.6, 258.7, 248.4,
                255.5, 240.4, 247.7, 249.1, 251.8, 262., 263.3, 280., 288.5,
                300.5, 312.6, 322.5, 324.3, 333.1, 344.8, 350.3, 369.1, 356.4])

D1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
               0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

D4 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 
               0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])


model = regression_model()
# Changing slope
#model.load_arrays(FRIG=FRIG, DUR=DUR, DURD1=DUR*D1, DURD4=DUR*D4)
# Changing intercept
model.load_arrays(FRIG=FRIG, DUR=DUR, D1=D1, D4=D4)
model.set_vars('FRIG', 'DUR', 'D1', 'D4')
model.fit_model()
params = model.get_params()
'''