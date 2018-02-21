import numpy as np
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