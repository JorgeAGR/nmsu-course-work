#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:04:01 2018

@author: jorgeagr
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

sns.set(rc={'figure.figsize':[width, height], 'axes.facecolor':'whitesmoke'})
#sns.set()
#sns.set_context('paper')

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)


class LSF(object):
    
    def __init__(self, x, y, sx, sy, fit='linear', n=1):
        
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
        self.fit = fit
        self.n = n # Should be 1 unless dealing with polys
        
        self.fit_poly()
        self.evaluate()
        
        print(self.a, self.da, self.rechisq)
    
    def fit_poly(self):
        # Data Vector (Outputs)
        X = np.zeros(shape = (self.n+1, 1))
        # Measurement Matrix (Inputs)
        M = np.zeros(shape = (self.n+1, self.n+1))
        # Coeffficients Vector
        self.a = np.zeros(self.n+1)
        counter = 0
        while True:
            counter += 1
            #dydx = np.zeros(len(x))
            A = self.a.copy()
            self.w = 1 / (self.sy ** 2 + (self.dfunc(self.x) * self.sx)**2)
            for i in range(self.n+1):
                for j in range(self.n+1):
                    M[i][j] = np.sum(self.w * self.x**(j+i))
                X[i][0] = np.sum(self.w * self.y * self.x ** i)
            self.a = np.dot(np.linalg.inv(M), X).flatten()
            if (np.abs(A - self.a).all() < 1e-12) or (counter == 100):
                break
        
        self.a = self.a.flatten()
        self.da = np.sqrt(np.linalg.inv(M).diagonal())
        
    def evaluate(self):
        chisq = np.sum((self.y - self.func(self.x))**2 * self.w) # chisq matters too!
        self.rechisq = chisq / (len(self.x) - 2) # Check this? Apparent DOF is not just on x & y
        
        #WSSR - Weighted sum square residuals
        
    def func(self, x):
        def linear():
            return self.a[0] + self.a[1] * x
        
        def poly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * x ** (i)
            return f
        
        def trig():
            return self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        funcs = {'linear': linear,
                'poly': poly,
                'trig': trig,}
        return funcs.get(self.fit)()
    
    def dfunc(self, x):
        def dlinear():
            return self.a[1]
        
        def dpoly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * i * x ** (i-1)
            return f
        
        def dtrig():
            return -self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        dfuncs = {'linear': dlinear,
                  'poly': dpoly,
                  'trig': dtrig,}
        return dfuncs.get(self.fit)()

x = np.array([0.0497, 0.1044, 0])# 0.1529, 0])
sx = np.array([0, 0, 0, ])
y = np.array([1.13, 1.14, 0.998])#1.05, 1.10])
sy = np.array([1, 1, 0.0000000001, ])

fit = LSF(x, y, sx, sy)
'''
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, fit.func(x))
'''
f = lambda x: (x - fit.a[0]) / fit.a[1]

x_pre = np.array([0.993, 1.006, 1.021, 1.030])
sx_pre = np.array([0, 0, 0, 0])
y_pre = np.array([0, 0.1138, 0.2275, 0.3322]) * 100
sy_pre = np.array([1, 1, 1, 1])

fit_pre = LSF(x_pre, y_pre, sx_pre, sy_pre)
'''
fig_pre, ax_pre = plt.subplots()
ax_pre.scatter(x_pre, y_pre)
ax_pre.plot(x_pre, fit_pre.func(x_pre))
'''

# Post-lab
fig, ax = plt.subplots()
fig.set_size_inches(width, height)
ax.set_title('Sugar', fontweight = 'bold')
#text = 'm = {:.2} \nb = {:.2}'.format(fit.a[1], fit.a[0])
#ax.text(0.08, 1.11, text, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})

df = pd.DataFrame(data = {'Density [g/mL]': y, 'Mass Percentage [%]': x})
sns.relplot(x='Mass Percentage [%]', y='Density [g/mL]', data=df, ax=ax)

df.loc[:, 'Density [g/mL]'] = fit.func(x)
sns.lineplot(x='Mass Percentage [%]', y='Density [g/mL]', data=df, ax=ax)


# Pre-lab
fig, ax = plt.subplots()
fig.set_size_inches(width, height)
ax.set_title('Ethylene Glycol', fontweight = 'bold')

df = pd.DataFrame(data = {'Density [g/mL]': x_pre, 'Mass Percentage [%]': y_pre})
sns.relplot(x='Density [g/mL]', y='Mass Percentage [%]', data=df, ax=ax)

df.loc[:, 'Mass Percentage [%]'] = fit_pre.func(x_pre)
sns.lineplot(x='Density [g/mL]', y='Mass Percentage [%]', data=df, ax=ax)


'''
x = np.array([0.497, 0.1044, 0.1529, 0])
sx = np.array([0, 0, 0, 0])
y = np.array([1.13, 1.14, 1.05, 0.998])
sy = np.array([1, 1, 1, 0.000000001])

fit = LSF(x, y, sx, sy)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, fit.func(x))

f = lambda x: (x - fit.a[0]) / fit.a[1]
'''
plt.show()