import numpy as np
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

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

L = ufloat(0.135,0.001) # m

d1nom = 0.123 # nm
d2nom = 0.213 # nm

V1 = ufloat(3,0.1) # kV
V2 = ufloat(4,0.1) # kV
V3 = ufloat(4.9,0.1) # kV

# measurements in cm
measurementSmallV1 = [
{'inner': 2.65 , 'outer': 3.19}, # Patrick
{'inner': 2.76, 'outer': 3.03}, # Ryan
{'inner': 2.51, 'outer': 2.95}, # Jorge
]

measurementLargeV1 = [
{'inner': 4.61, 'outer': 5.19}, # Patrick
{'inner': 4.75, 'outer': 5.14}, # Ryan
{'inner': 4.87, 'outer': 5.18}, # Jorge
]

measurementSmallV2 = [
{'inner': 2.30, 'outer': 2.69}, # Patrick
{'inner': 2.33, 'outer': 2.65}, # Ryan
{'inner': 2.34, 'outer': 2.72}, # Jorge
]

measurementLargeV2 = [
{'inner': 3.99, 'outer': 4.46}, # Patrick
{'inner': 4.14, 'outer': 4.49}, # Ryan
{'inner': 4.12, 'outer': 4.45}, # Jorge
]

measurementSmallV3 = [
{'inner': 2.15, 'outer': 2.50}, # Patrick
{'inner': 2.09, 'outer': 2.39}, # Ryan
{'inner': 2.23, 'outer': 2.49},# Jorge
]

measurementLargeV3 = [
{'inner': 3.67, 'outer': 3.99}, # Patrick
{'inner': 3.70, 'outer': 4.01}, # Ryan
{'inner': 3.68, 'outer': 4.05}, # Jorge
]

        
smallV1_avg = avgMeasurement(groupMeasurement(measurementSmallV1), 0.1, 1)
largeV1_avg = avgMeasurement(groupMeasurement(measurementLargeV1), 0.1, 1)

smallV2_avg = avgMeasurement(groupMeasurement(measurementSmallV2), 0.1, 1)
largeV2_avg = avgMeasurement(groupMeasurement(measurementLargeV2), 0.1, 1)

smallV3_avg = avgMeasurement(groupMeasurement(measurementSmallV3), 0.1, 1)
largeV3_avg = avgMeasurement(groupMeasurement(measurementLargeV3), 0.1, 1)

d2_V1 = dfunction(L, smallV1_avg, V1)
d1_V1 = dfunction(L, largeV1_avg, V1)
d2_V2 = dfunction(L, smallV2_avg, V2)
d1_V2 = dfunction(L, largeV2_avg, V2)
d2_V3 = dfunction(L, smallV3_avg, V3)
d1_V3 = dfunction(L, largeV3_avg, V3)

d1 = (d1_V1.n, d1_V2.n, d1_V3.n)
d1_s = (d1_V1.s, d1_V2.s, d1_V3.s)
d2 = (d2_V1.n, d2_V2.n, d2_V3.n)
d2_s = (d2_V1.s, d2_V2.s, d2_V3.s)
V = (V1.n, V2.n, V3.n)
V_s = (V1.s, V2.s, V3.s)

d1_avg = np.mean(d1)
d2_avg = np.mean(d2)

fig1, ax1 = plt.subplots()
ax1.errorbar(d1, V, xerr = d1_s, yerr = V_s, fmt = 'o', label = '$d_1$ data')
ax1.errorbar(d2, V, xerr = d2_s, yerr = V_s, fmt = 'o', label = '$d_2$ data')
ax1.set_title('Calculated diffraction slits')
ax1.set_xlabel('Calculated Slit Size $d_n$ $(nm)$')
ax1.set_ylabel('Voltage (kV)')
ax1.grid()
ax1.legend()