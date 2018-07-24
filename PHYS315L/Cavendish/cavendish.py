# == Rutherford Scattering == #
import numpy as np
import csv
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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

def in2m(x):
    f = 0.0254 * x
    return f

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]


# == Voltage - Angle Relationship == #

'''
Volts to Angle:
    Init (0):
        I: 0 +- 0.001 A
        V: 266.8287 +- 37.6632 microV
        Theta: 3 +- 0.1 rad
Left:
    Jorge:
        0.02 rad change
        I: 0.038 +- 0.001 A
        V: -44.22759 +- 0.07881 mV
        
        0.1 rad change
        I: 0.151 +- 0.001 A
        V: -129.0290 +- 72.2142 mV
        
        0.5 rad change
        I: 0.849 +- 0.001 A
        V: -887.7279 +- 87.4721 mV
    
    Patrick:
        0.02 rad change
        I: 0.031 +- 0.001 A
        V: -35.46271 +- 0.162648 mV
        
        0.1 rad change
        I: 0.155 +- 0.001 A
        V: -181.5122 +- 0.075526 mV
        
        0.5 rad change
        I: 0.836 +- 0.001 A
        V: -875.5364 +- 0.0565268 mV
    
    Ryan:
        0.02 rad change
        I: 0.029 +- 0.001 A
        V: -35.65744 +- 0.0706265 mV
        
        0.1 rad change
        I: 0.145 +- 0.001
        V: -170.9361 +- 0.101584 mV
        
        0.5 rad change
        I: 0.836 +- 0.001 A
        V: -875.4954 +- 0.189742 mV

Right:
    Jorge:
        0.02 rad change
        I: 0.034 +- 0.001 A
        V: 36.87462 +- 0.0551895 mV
        
        0.1 rad change
        I: 0.150 +- 0.001 A
        V: 171.4923 +- 0.0750963 mV
        
        0.5 rad change
        I: 0.850 +- 0.001 A
        V: 865.5919 +- 0.0376274 mV
        
    Patrick:
        0.02 rad change
        I: 0.027 +- 0.001 A
        V: 31.12234 +- 0.184752 mV
        
        0.1 rad change
        I: 0.144 +- 0.001 A
        V: 166.1184 +- 0.0246552 mV
        
        0.5 rad change
        I: 0.850 +- 0.001 A
        V: 865.3777 +- 0.0374929 mV
    
    Ryan:
        0.02 rad change
        I: 0.032 +- 0.001 A
        V: 35.40876 +- 0.0549347 mV
        
        0.1 rad change
        I: 0.153 +- 0.001 A
        V: 176.3709 +- 0.0616682 mV
        
        0.5 rad change
        I: 0.858 +- 0.001 A
        V: 870.7073 +- 0.0741271 mV
'''

n_angle_to_volts = {0.02: [-44.22759, -35.46271, -35.65744, 36.87462, 31.12234, 35.40876],
                 0.1: [-129.0290, -181.5122, -170.9361, 171.4923, 166.1184, 176.3709],
                 0.5: [-887.7279, -875.4954, -875.4954, 865.5919, 865.3777, 870.7073]}

s_angle_to_volts = {0.02: [0.07881, 0.162648, 0.0706265, 0.0551895, 0.184752, 0.0549347],
                   0.1: [72.2142, 0.075526, 0.101584, 0.0750963, 0.0246552, 0.0616682],
                   0.5: [87.4721, 0.0565268, 0.189742, 0.0376274, 0.0374929, 0.0741271]}

volts = unumpy.uarray(np.abs([-44.22759, -35.46271, -35.65744, 36.87462, 31.12234, 35.40876,
                        -129.0290, -181.5122, -170.9361, 171.4923, 166.1184, 176.3709,
                        -887.7279, -875.4954, -875.4954, 865.5919, 865.3777, 870.7073]),
                        [0.07881, 0.162648, 0.0706265, 0.0551895, 0.184752, 0.0549347,
                         72.2142, 0.075526, 0.101584, 0.0750963, 0.0246552, 0.0616682,
                         87.4721, 0.0565268, 0.189742, 0.0376274, 0.0374929, 0.0741271]) * 1e-3

radians = unumpy.uarray([0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
                         0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                         0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                         0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

anglefit, anglepopt, angleperr, anglechisq, anglechire = linearLSF(unumpy.nominal_values(volts),
                                                                   unumpy.nominal_values(radians),
                                                                   unumpy.std_devs(volts))

voltfit, voltpopt, voltperr, voltchisq, voltchire = linearLSF(unumpy.nominal_values(radians),
                                                              unumpy.nominal_values(volts),
                                                              unumpy.std_devs(volts))

def volt2rad_0(x):
    f = anglepopt[1] * x + anglepopt[0]
    return f


def volt2rad(x):
    f = x/voltpopt[1] - voltpopt[0]/voltpopt[1]
    return f

fig, ax = plt.subplots()
fig.suptitle('Voltage - Angle Relationship', fontweight = 'bold')
ax.errorbar(unumpy.nominal_values(radians),
            unumpy.nominal_values(volts),
            xerr = unumpy.std_devs(radians), yerr = unumpy.std_devs(volts),
            fmt = '.', label = 'Measured Data')
ax.plot(unumpy.nominal_values(radians), voltfit, label = 'LLS Fit')
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.set_ylabel(r'Voltage $[V]$')
ax.set_xlabel(r'Radians $[rad]$')
ax.grid()
ax.legend()

# == Torsion Coefficient (K) and Inertia == #
'''
    0 Brass:
        Init: 0.500 s
        Final: 1.650 s
    
    2 Brass:
        Init: 0.950 s
        Final: 2.250 s
    
    4 Brass:
        Init: 0.750 s
        Final: 2.250 s
    
    6 Brass:
        Init: 1.300 s
        Final: 2.850 s
    
    8 Brass:
        Init: 1.800 s
        Final: 3.500 s
'''

periods = unumpy.uarray([1.650 - 0.500, 2.250 - 0.950, 2.250 - 0.750, 2.850 - 1.300, 3.500 - 1.800], 
                        [0.05, 0.05, 0.05, 0.05, 0.05])

n = np.array([0, 2, 4, 6, 8])

pfit, ppopt, pperr, pchisq, pchire = linearLSF(n, unumpy.nominal_values(periods), unumpy.std_devs(periods))

mbrass = ufloat(214.5, 0.5) * 1e-3
I_r1 = in2m(0.86)
I_r2 = in2m(1.86)

dI = (1/2) * mbrass * (I_r1**2 + I_r2**2)

K = 4 * (np.pi**2) * dI / ppopt[1]

I0 = ppopt[0] * K / (4 * np.pi**2)

Lrod = ufloat(91.7, 0.1) * 1e-2

mrod = ufloat(241.3, 0.1) * 1e-3

murod = mrod / Lrod

# B - C \ A - D

mB = ufloat(167.1469, 1.1e-3) * 1e-3 + murod * 2.91e-2
mA = ufloat(164.5211, 1.1e-3) * 1e-3 + murod * 2.91e-2

MC = ufloat(4279.259, 30e-3) * 1e-3
MD = ufloat(4281.787, 30e-3) * 1e-3

rAD = 24.6e-3 + 44.5e-3
rBC = 27.0e-3 + 44.5e-3

noballs_v = unumpy.uarray([4.648660, 4.349725, 4.535125, 4.884205, 4.820535], 
                        [0.0182331, 0.0167990, 0.0158124, 0.0135203, 0.0195627]) * 1e-3

balls_v = unumpy.uarray([6.436821, 5.253742, 5.737554, 5.868733, 5.466052], 
                      [0.0176908, 0.0188325, 0.0196152, 0.0159878, 0.0168731]) * 1e-3

noballs = volt2rad(noballs_v)

balls = volt2rad(balls_v)

theta = balls - noballs

def Gconst_simple(K, theta, r, L, m, M):
    
    G = (K * theta * r**2) / (L * m * M)
    return G

def Gconst(K, theta, r1, r2, L, m1, m2, M1, M2):
    
    G = (2 * K * theta) / ((M1 * m1 / r1**2) + (M2 * m2 / r2**2)) / L
    return G

gconst = Gconst(K, theta, rAD, rBC, Lrod, mA, mB, MD, MC)

fig2, ax2 = plt.subplots()
fig2.suptitle('Torsion Coefficient and Moment of Inertia', fontweight = 'bold')
ax2.errorbar(n, unumpy.nominal_values(periods), yerr = unumpy.std_devs(periods), fmt = '.', label = 'Measured Data')
ax2.plot(n, pfit, label = 'LLS Fit')
ax2.set_xlabel('Number of brass quadrant masses')
ax2.set_ylabel('Period $[s]$')
ax2.grid()
ax2.legend()

fig3, ax3 = plt.subplots()
fig3.suptitle('Measured Gravitational Constant', fontweight = 'bold')
ax3.set_title(r'Shown error bars are $20\sigma$')
ax3.errorbar(range(1,6), unumpy.nominal_values(gconst), yerr = 20*unumpy.std_devs(gconst),fmt = '.', label = 'Calculated Values')
ax3.axhline(6.676e-11, color = 'black')
ax3.annotate('Nominal Value', xy=(1, 7e-11), xytext=(1, 7e-11))
ax3.xaxis.set_major_locator(MultipleLocator(1))
ax3.set_ylim(1e-11, 1e-5)
ax3.set_yscale('log')
ax3.set_ylabel(r'Gravitational Constant $[\frac{m^3}{kg \cdot s^2}]$')
ax3.set_xlabel('Trial #')
ax3.grid()
ax3.legend()

'''
Patrick: 27mm + (44.5) (B-C)
Ryan: 24.60625 + (44.5) (A-D)
Main Show:
    NoBalls: -4.648660 +- 0.0182331
    Balls: -6.436821 +- 0.0176908
    
    NoBalls: -4.349725 +- 0.0167990
    Balls: -5.253742 +- 0.0188325
    
    NoBalls: -4.535125 +- 0.0158124
    Balls: -5.737554 +- 0.0196152
    
    NoBalls: -4.884205 +- 0.0135203
    Balls: -5.868733 +- 0.0159878
    
    NoBalls: -4.820535 +- 0.0195627
    Balls: -5.466052 +- 0.0168731
'''