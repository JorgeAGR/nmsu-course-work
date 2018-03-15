import numpy as np
import csv
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

def shiftMeasurement(measurement, center):
    shift = measurement[center]
    for i in measurement:
        measurement[i] = np.abs(measurement[i] - shift)

def avgMeasurement(measurement, instrumental_error, decimal_round):
    avgM = np.mean(measurement)
    errorM = np.std(measurement)
    if (instrumental_error > errorM):
        errorM = instrumental_error
    avgR = round(avgM, decimal_round)
    errorR = round(errorM, decimal_round)
    avg = ufloat(avgR, errorR)
    return avg

def linearLSF(x, y, dy):
    def linear(x, a, b):
        return b * x + a
    
    popt, pcov = curve_fit(linear, x, y)
    
    fit = linear(x, *popt)
    
    chisq = np.sum(((y - fit)/dy)**2) #Chi Squared
    chire = chisq / (len(x) - len(popt)) # Chi-Reduced (Chi Squared / Dof)
    perr = np.sqrt(np.diag(pcov)) # Error in parameters a, b
    
    return fit, popt, perr, chisq, chire

h = 6.626e-34
mu = 5.051e-27

''' Current: 3.48 A
        Patrick: 434 mT
        Ryan: 436 mT
        Jorge: 436 mT
        
        #Finals are with sample center
        #Initials are with sample 1 grid (5 units) to the right
        #Parenthesis next to measurement are for FWHM
        
        vof = 18.4258 (1 unit) - Water
        voi = 18.4358MHz
        
        vuf = 18.4265 (0.5 units) - Polystyrene
        vui = 18.4385
        
        vcf = 18.4264 (1.5 units) - Glycerine
        vci = 18.4373
        
        vwf = 17.3383 (2 units) -Teflon
        vwi = 17.3490
'''
''' Current: 4.00 A
        Patrick: 467 mT
        Ryan: 463
        Jorge: 464
        
        vof = 19.8561 (0.5 units)
        voi = 19.8674
        
        vcf = 19.8560 (1 units)
        vci = 19.8671
        
        vuf = 19.8564 (0.5 units)
        vui = 19.8662
        
        vwf = 18.6807 (2 units)
        vwi = 18.6907
'''
''' Current: 3.75 A
        Patrick:  446 mT
        Ryan:   448 mT
        Jorge: 446 mT
        
        vof = 19.1898 (1 unit)
        voi = 19.1981
        
        vcf = 19.1877(1 unit)
        vci = 19.1977
        
        vuf = 19.1860 (1 unit)
        vui = 19.1963
        
        vwf = 18.0480 (2 units)
        vwi = 18.0572
'''