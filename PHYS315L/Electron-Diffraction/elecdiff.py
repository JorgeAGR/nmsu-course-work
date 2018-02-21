import numpy as np
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

def groupMeasurement(measurementDic):
    marray = []
    for m in measurementDic:
        for c in m:
            marray.append(m[c])
    return marray

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

def dfunction(Length, Diameter, Voltage):
    d = (2.46 * Length) / (Diameter * 10**(-2) * umath.sqrt(Voltage * 10**(3)))
    return d

L = ufloat(0.132,0.001) # m

d1 = 0.123 # nm
d2 = 0.213 # nm

V1 = ufloat(3,0.1) # kV
V2 = ufloat(4,0.1) # kV
V3 = ufloat(4.9,0.1) # kV

# measurements in cm
measurementSmallV1 = [
{'inner': 3.89 , 'outer': 3.95}, # Patrick 1
{'inner': 3.51, 'outer': 3.86}, # Ryan 1
{'inner': 3.49, 'outer': 3.87}, # Patrick 2
{'inner': 3.46, 'outer': 3.82}, # Ryan 2
]

measurementLargeV1 = [
{'inner': 5.33 , 'outer': 5.82}, # Patrick 1
{'inner': 5.93, 'outer': 6.02}, # Ryan 1
{'inner': 5.39, 'outer': 5.95}, # Patrick 2
{'inner': 5.50, 'outer': 5.87}, # Ryan 2
]

measurementSmallV2 = [
{'inner': 3.12, 'outer': 3.45}, # Patrick 1
{'inner': 3.12, 'outer': 3.43}, # Ryan 1
{'inner': 3.13, 'outer': 3.39}, # Patrick 2
{'inner': 3.12, 'outer': 3.38}, # Ryan 2
]

measurementLargeV2 = [
{'inner': 4.75, 'outer': 5.19}, # Patrick 1
{'inner': 4.81, 'outer': 5.14}, # Ryan 1
{'inner': 4.70, 'outer': 5.19}, # Patrick
{'inner': 4.82, 'outer': 5.18}, # Ryan 2
]

measurementSmallV3 = [
{'inner': 2.99, 'outer': 3.23}, # Patrick 1
{'inner': 2.91, 'outer': 3.18}, # Ryan 1
{'inner': 2.94, 'outer': 3.23},# Patrick 2
{'inner': 2.94, 'outer': 3.19}, # Ryan 2
]

measurementLargeV3 = [
{'inner': 4.45, 'outer': 4.85}, # Patrick 1
{'inner': 4.49, 'outer': 4.82}, # Ryan 1
{'inner': 4.36, 'outer': 4.70}, # Patrick 2
{'inner': 4.50, 'outer': 4.81}, # Ryan 2
]

        
smallV1_avg = avgMeasurement(groupMeasurement(measurementSmallV1), 0.1, 1)
largeV1_avg = avgMeasurement(groupMeasurement(measurementLargeV1), 0.1, 1)

smallV2_avg = avgMeasurement(groupMeasurement(measurementSmallV2), 0.1, 1)
largeV2_avg = avgMeasurement(groupMeasurement(measurementLargeV2), 0.1, 1)

smallV3_avg = avgMeasurement(groupMeasurement(measurementSmallV3), 0.1, 1)
largeV3_avg = avgMeasurement(groupMeasurement(measurementLargeV3), 0.1, 1)

d1_V1 = dfunction(L, smallV1_avg, V1)
d2_V1 = dfunction(L, largeV1_avg, V1)
d1_V2 = dfunction(L, smallV2_avg, V2)
d2_V2 = dfunction(L, largeV2_avg, V2)
d1_V3 = dfunction(L, smallV3_avg, V3)
d2_V3 = dfunction(L, largeV3_avg, V3)