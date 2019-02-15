#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:17:32 2018

@author: jorgeagr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.stats import chi2

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

def efficency(success, total):
    e = success/total
    sig_e = np.sqrt(e * (1 - e) / total)
    return e, sig_e

class LSF(object):
    
    def __init__(self, x, y, sx, sy, fit='poly', n=1):
        
        self.x = np.asarray(x, dtype=np.double)
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
        self.chisq = np.sum((self.y - self.func(self.x))**2 * self.w) # chisq matters too!
        self.rechisq = self.chisq / (len(self.x) - self.n + 1) # Check this? Apparent DOF is not just on x & y
        
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


# Top scintillators

def top_scintillators():
    voltage = np.array([1550, 1560, 1565, 1570, 1580])
    
    counts = np.array([65, 74, 77, 57, 56])
    #1565 V
    top_singles = 991 #cpm
    bottom_singles = 4023 #cpm
    
    used = np.where(voltage == 1565)
    
    fig1, ax0 = plt.subplots()
    
    ax0.errorbar(voltage, counts, yerr = np.sqrt(counts), fmt = '.', color='black')
    ax0.errorbar(voltage[used], counts[used], yerr = np.sqrt(counts[used]), 
                 fmt='.', color='red')
    ax0.grid()
    ax0.set_title('Top Scintillators Counts', weight='bold')
    ax0.set_xlabel('Voltage [V]')
    ax0.set_ylabel('Counts [1/min]')
    plt.tight_layout()


# Efficency Barrel
def barrel():
    voltage_barrel = np.array([1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 
                               1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075,
                               2100, 2125, 2150, 2175, 2200])
    
    counts_barrel = np.array([17, 13, 17, 21, 25, 34, 29, 36, 
                              42, 40, 39, 37, 47, 62, 51, 52,
                              51, 50, 57, 58, 56])
    
    counts_total = np.array([78, 64, 62, 67, 64, 72, 49, 66, 
                             72, 62, 63, 56, 61, 84, 68, 71,
                             72, 62, 65, 72, 67])
        
    singles = np.array([3956,6358,9524,13192,17192,22713,28245,35787,
                        44169,59170,76794,102284,134541,175262,212746,
                        262682,340001,423007,495895,595958,657816])
    
    eff_barrel, sig_eff_barrel = efficency(counts_barrel, counts_total)
    
    used = np.where(voltage_barrel == 1800)
    
    fig1, ax1 = plt.subplots(nrows = 2)
    #fig1.set_size_inches(9,15)
    
    ax1[0].errorbar(voltage_barrel, counts_barrel, yerr = np.sqrt(counts_barrel), 
       color='black', fmt = '.')
    ax1[0].errorbar(voltage_barrel[used], counts_barrel[used], yerr = np.sqrt(counts_barrel[used]),
       color='red', fmt='.')
    ax1[0].set_xlabel('Voltage [V]')
    ax1[0].set_ylabel('Counts [1/min]')
    ax1[0].grid()
    ax1[0].set_title('Barrel Counts', weight='bold')
    
    ax1[1].errorbar(voltage_barrel, eff_barrel * 100, yerr = sig_eff_barrel * 100, 
       color='black', fmt = '.')
    ax1[1].errorbar(voltage_barrel[used], eff_barrel[used] * 100,
       sig_eff_barrel[used] * 100, color='red', fmt='.')
    ax1[1].set_xlabel('Voltage [V]')
    ax1[1].set_ylabel('Efficiency [%]')
    ax1[1].grid()
    ax1[1].set_title('Barrel Efficency', weight='bold')
    plt.tight_layout()
    
    fig, ax = plt.subplots()
    ax.errorbar(voltage_barrel, singles/6, yerr = np.sqrt(singles)/6, 
       color='black', fmt = '.')
    ax.errorbar(voltage_barrel[used], singles[used]/6, yerr = np.sqrt(singles[used])/6,
       color='red', fmt='.')
    ax.set_yscale('log')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Counts [1/s]')
    ax.grid()
    ax.set_title('Barrel Singles', weight='bold')
    plt.tight_layout()

'''
Have to stay below 10^4 Hz (10 KHz), so voltage ~ 1900 V
'''

# Efficency Vetos
def veto():
    voltage_barrel = np.array([1800, 1825, 1850, 1875, 
                           1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075,
                           2100, 2125, 2150])
    
    voltage_veto = voltage_barrel + 100
    
    counts_4co = np.array([6, 4, 5, 5, 10, 11, 7, 16, 7, 6, 6, 9, 3, 9, 3])
    
    counts_3co = np.array([43, 33, 43, 37, 45, 48, 55, 48, 45, 45, 38, 42, 35, 52, 27])
    
    singles = {'2100 V': 460} #cps
    
    eff_veto, sig_eff_veto = efficency(counts_4co, counts_3co)
    
    fig2, ax2 = plt.subplots(nrows = 2)
    #fig2.set_size_inches(9, 9)
    
    used = np.where(voltage_veto == 2100)
    
    ax2[0].errorbar(voltage_veto, counts_4co, yerr = np.sqrt(counts_4co), 
       fmt='.', color='black')
    ax2[0].errorbar(voltage_veto[used], counts_4co[used],
       yerr = np.sqrt(counts_4co[used]), fmt='.', color='red')
    ax2[0].grid()
    ax2[0].set_title('Veto Counts', weight='bold')
    ax2[0].set_ylabel('Counts [1/min]')
    ax2[0].set_xlabel('Voltage [V]')
    
    ax2[1].errorbar(voltage_veto[0:len(eff_veto)], 
       eff_veto * 100, yerr = sig_eff_veto * 100, fmt = '.', color='black')
    ax2[1].errorbar(voltage_veto[used], eff_veto[used] * 100,
       yerr = sig_eff_veto[used] * 100, fmt='.', color='red')
    ax2[1].grid()
    ax2[1].set_title('Veto Efficency', weight='bold')
    ax2[1].set_ylabel('Efficency [%]')
    ax2[1].set_xlabel('Voltage [V]')
    plt.tight_layout()

# Time calibration (in microsec)
# Nominal delays: 20, 15, 10, 5, 2
def time():
    delay = np.array([19.40, 14.00, 8.96, 5.28, 2.050])
    delay_s = np.array([0.05, 0.05, 0.02, 0.02, 0.05])
    channel = np.array([646, 459, 286, 159, 48])
    
    fig3, ax3 = plt.subplots()
    #fig3.set_size_inches(9, 4.5)
    # 10sigma error bars for reference
    ax3.errorbar(channel, delay, yerr = 10*delay_s, fmt = 'o', 
                 color='gray', label='Measured Delays')
    ax3.grid()
    ax3.set_title('Channel vs Time', weight='bold')
    
    ch2time = LSF(channel, delay, np.zeros(len(channel)), delay_s, n=1)
    ax3.plot(range(48, 647), ch2time.func(np.arange(48, 647)),
             color='black', linewidth = 1, label='Fit')
    ax3.legend()
    ax3.set_xlabel('Channel')
    ax3.set_ylabel(r'Time [$\mu$s]')
    plt.tight_layout()
    #print('Factor:', ch2time.a[1])
    return ch2time

def muon():
    
    def f(x, A, t, C):
        return A * np.exp(-t * x) + C
    
    def linear(x, A, t, B, C):
        return A * np.exp(-t * x) + B * x + C
    
    muon_counts = np.loadtxt('muon_counts.txt')
    muon_counts = muon_counts[muon_counts > 0]
    channels = np.arange(len(muon_counts)) + 1
    
    ch2time = time()
    
    # =====
    
    param, error = curve_fit(f, channels, muon_counts, sigma=np.sqrt(muon_counts), 
                             p0=[muon_counts[0], 1, muon_counts[-1]])
    error = np.sqrt(np.diag(error))
    
    halflife = ch2time.a[1] / param[1]
    halflife_s = halflife * np.sqrt( (ch2time.da[1]/ch2time.a[1])**2 + (error[1]/param[1])**2 )
    
    dof = len(muon_counts) - len(param)
    chisq = np.sum(( (f(channels, param[0], param[1], param[2]) - muon_counts) / np.sqrt(muon_counts) ) ** 2)
    rechisq = chisq / dof
    pval = chi2.sf(chisq, dof) * 100
    
    
    #print(param[1], error[1])
    print('\nConstant Fit:')
    print('Decay Const (in ch):', param[1], '+/-', error[1])
    print('Half-life:', halflife, '+/-', halflife_s)
    print('dT/T:', halflife_s/halflife)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq per dof:', rechisq)
    print('p-value:', pval, '%')
    
    fig4, ax4 = plt.subplots()
    #fig4.set_size_inches(9, 4.5)
    ax4.errorbar(channels, muon_counts, yerr= np.sqrt(muon_counts), fmt='.', 
                 color='gray', label='Data')
    ax4.plot(channels, f(channels, param[0], param[1], param[2]), color='black', 
             linewidth=2.5, label='Constant bg')
    ax4.grid()
    ax4.set_xlim(0,channels[-1])
    ax4.set_title('Captured Muons', weight='bold')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Frequency')
    plt.tight_layout()
    
    # =====
    
    param, error = curve_fit(linear, channels, muon_counts, sigma=np.sqrt(muon_counts), 
                             p0=[muon_counts[0], 1, muon_counts[-1], muon_counts[-1]])
    error = np.sqrt(np.diag(error))
    
    halflife = ch2time.a[1] / param[1]
    halflife_s = halflife * np.sqrt( (ch2time.da[1]/ch2time.a[1])**2 + (error[1]/param[1])**2 )
    
    dof = len(muon_counts) - len(param)
    chisq = np.sum(( (linear(channels, param[0], param[1], param[2], param[3]) - muon_counts) / np.sqrt(muon_counts) ) ** 2)
    rechisq = chisq / (len(muon_counts) - len(param))
    pval = chi2.sf(chisq, dof) * 100
    
    print('\nLinear Fit:')
    print('Decay Const (in ch):', param[1], '+/-', error[1])
    print('Half-life:', halflife, '+/-', halflife_s)
    print('dT/T:', halflife_s/halflife)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq per dof:', rechisq)
    print('p-value:', pval, '%')
    print('\n')
    
    ax4.plot(channels, linear(channels, param[0], param[1], param[2], param[3]), color='red', 
             linewidth=2.5, label='Linear bg')
    ax4.legend()
    
    '''
    fig, ax = plt.subplots()
    ax.errorbar(channels, muon_counts, yerr= np.sqrt(muon_counts), fmt='.', 
                 color='gray', label='Data')
    ax.grid()
    ax.legend()
    ax.set_title('Captured Muons', weight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    '''
    
def main():
    top_scintillators()
    barrel()
    veto()
    muon()
# Preliminary calculations
# From fityk:
# const = 45.1746 +/- 0.43742
# amp = 29.0299 +/- 1.67317
# decay = 102.682 +/- 10.0514
# halflife = ch2time.a[1] * 102.682
# halflife_s = np.sqrt((102.682)**2 * (ch2time.da[1])**2 + (0.029)**2 * (10.05)**2)