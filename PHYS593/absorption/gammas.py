# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:55:56 2018

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import interpolate

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

mc2 = 0.5109989461 # MeV
q =  1.6021766208e-19
E_sr90 = 545.86e-3
E_yt91 = 2.2801

def decay(x, A, k):
        return A * np.exp(-k * x)

def aluminum():
    al_thickness = np.array([0, 3.0, 6.7, 13.6, 20.4,34.7, 82.1,
                        111.9, 135.6, 171.6, 223.4,277.0,343.4, 437.1,
                        491.7,547.8,631.2, 679.7, 836.2, 960.5,1086.8,
                        1213.3,1372.8,1491, 1605.8]) * 1e-3 # g/cm^2
    
    al_height = np.array([1781.53, 1924.64, 1931.65, 1922.55, 1940.86, 1921.2, 1915.57,
                         1909.77, 1913.99, 1904.11, 1896.56,1902.69,1863.86, 1874.84,
                         1862.22,1852.36,1853.41, 1832.69, 1809.91, 1788.54, 1784.45,
                         1755.23,1748.88,1730.02, 1726.81  ])
    al_height_s = np.array([9.98743, 9.01467, 8.81117, 9.38228, 9.30487, 8.8604, 7.75737,
                            9.0464, 7.81719, 9.18021, 8.52465,7.51524,8.40789,8.74458,
                            8.2597,8.10565,8.59083,8.59043, 8.26304, 8.77227,8.26287,
                            8.59499,8.37597,8.30718, 8.59852  ])
     
    al_hwhm = np.array([34.929, 32.39601, 32.2375, 32.321, 32.1898, 32.3934, 32.5623,
                        32.3741, 32.8108, 32.0552, 32.1729,32.6063,32.0974,32.1239, 
                        32.4123,32.4082,31.9771, 32.3521, 32.4151,32.4797,32.2316,  
                        32.3068,32.6554,32.2672, 32.2652 ])
    al_hwhm_s = np.array([0.306901, 0.246315, 0.240202, 0.256317, 0.251162, 0.243825, 0.213755,
                           0.29643, 0.215646, 0.254029, 0.23671,0.200769,0.236887,0.245407,
                           0.234622,0.23121,0.244225, 0.248235, 0.241331, 0.259291, 0.244482, 
                           0.259032,0.252989,0.254304, 0.262882])
    
    al_sigma = al_hwhm / np.sqrt(2 * np.log(2))
    al_sigma_s = al_hwhm_s / np.sqrt(2 * np.log(2))
    
    al_area = np.sqrt(2 * np.pi) * al_sigma * al_height
    al_area_s = al_area * np.sqrt((al_sigma_s / al_sigma)**2 + (al_height_s / al_height)**2)
    
    param, error = curve_fit(decay, al_thickness, al_area, sigma=al_sigma_s, 
                                 p0=[al_area[0], 0.001])
    error = np.sqrt(np.diag(error))
    
    print('Aluminum Results:')
    print('Coefficient of Absorption:', param[1], '+/-', error[1], 'cm^2/g')
    chisq = np.sum(( (decay(al_thickness, param[0], param[1]) - al_area) / al_area_s ) ** 2)
    dof = len(al_area) - len(param)
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    print('Fit Sr90 GOF Test:')
    print('Parameters:', param)
    print('Errors:', error)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq/dof:', rechisq)
    print('p-value:', pval, '%')
    
    normalize = decay(0, param[0], param[1])
    al_area = al_area / normalize
    al_area_s = al_area_s / normalize
    
    thick_grid = np.linspace(0, 2000e-3)
    fig, ax = plt.subplots()
    ax.errorbar(al_thickness, al_area, yerr=al_area_s, fmt='.', color='black', label='Data')
    ax.plot(thick_grid, decay(thick_grid, param[0], param[1])/normalize, '--', color='gray', label='Fit')
    ax.set_xlim(-0.0, 1.68)
    ax.set_ylim(0.88, 1.016)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.004))
    ax.set_xlabel(r'Thickness [mg/cm$^2$]')
    ax.set_ylabel('Intensity [%]')
    ax.legend()
    plt.tight_layout()

def lead():
    pb_thickness = np.array([0, 926.1, 1821.3, 2650.7, 4490.1, 7143.8]) * 1e-3
    
    pb_height = np.array([1781.53, 1757.8, 1609.14,1512.02, 1263.53, 997.205])
    pb_height_s = np.array([9.98743, 8.56149, 8.18373,6.95331, 6.15067, 5.54297])
    
    pb_hwhm = np.array([34.929, 31.9684, 32.3812, 32.3047, 32.2881, 32.9302])
    pb_hwhm_s = np.array([0.306901, 0.255552, 0.269567,0.24462, 0.259639, 0.301249])
    
    pb_sigma = pb_hwhm / np.sqrt(2 * np.log(2))
    pb_sigma_s = pb_hwhm_s / np.sqrt(2 * np.log(2))
    
    pb_area = np.sqrt(2 * np.pi) * pb_sigma * pb_height
    pb_area_s = pb_area * np.sqrt((pb_sigma_s / pb_sigma)**2 + (pb_height_s / pb_height)**2)
    
    param, error = curve_fit(decay, pb_thickness, pb_area, sigma=pb_sigma_s, 
                                 p0=[pb_area[0], 0.001])
    error = np.sqrt(np.diag(error))
    
    print('Lead Results:')
    print('Coefficient of Absorption:', param[1], '+/-', error[1], 'cm^2/g')
    chisq = np.sum(( (decay(pb_thickness, param[0], param[1]) - pb_area) / pb_area_s ) ** 2)
    dof = len(pb_area) - len(param)
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    print('Fit Sr90 GOF Test:')
    print('Parameters:', param)
    print('Errors:', error)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq/dof:', rechisq)
    print('p-value:', pval, '%')
    
    normalize = decay(0, param[0], param[1])
    pb_area = pb_area / normalize
    pb_area_s = pb_area_s / normalize
    
    thick_grid = np.linspace(0, 7500e-3)
    fig, ax = plt.subplots()
    ax.errorbar(pb_thickness, pb_area, yerr=pb_area_s, fmt='.', color='black', label='Data')
    ax.plot(thick_grid, decay(thick_grid, param[0], param[1])/normalize, '--', color='gray', label='Fit')
    ax.set_xlim(-0.1, 7250e-3)
    ax.set_ylim(0.5, 1.04)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
    ax.set_xlabel(r'Thickness [mg/cm$^2$]')
    ax.set_ylabel('Intensity [%]')
    ax.legend()
    plt.tight_layout()