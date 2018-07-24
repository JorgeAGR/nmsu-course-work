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

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

h = 6.626e-34
mu = 5.051e-27

''' Current: 3.48 A
        Patrick: 434 mT +- 1
        Ryan: 436 mT
        Jorge: 436 mT
        
        #Finals are with sample center
        #Initials are with sample 1 grid (5 units) to the right
        #Parenthesis next to measurement are for FWHM
        
        vof = 18.4258 (1 unit) - Water
        voi = 18.4358MHz
        
        vcf = 18.4264 (1.5 units) - Glycerine
        vci = 18.4373
        
        vuf = 18.4265 (0.5 units) - Polystyrene
        vui = 18.4385
        
        vwf = 17.3383 (2 units) -Teflon
        vwi = 17.3490
'''
# Sample Array Centered = Water, Glycerine, Polystyrene, Teflon
# Sample Array Box 1 = Water, Glycerine, Polystyrene, Teflon
b1_avg = np.mean([434, 436, 436])
db1_avg = np.std([434, 436, 436])

freqs_c1 = np.asarray([18.4258, 18.4264, 18.4265, 17.3383])
freqs_c1b = np.asarray([18.4358, 18.4373, 18.4385, 17.3490])

dfreqs_c1 = (np.asarray([1, 1.5, 0.5, 2]) * (freqs_c1b - freqs_c1)) / (2*np.sqrt(2*np.log(2)))

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
b2_avg = np.mean([446, 448, 446])
db2_avg = np.std([446, 448, 446])

freqs_c2 = np.asarray([19.1898, 19.1877, 19.1860, 18.0480])
freqs_c2b = np.asarray([19.1981, 19.1977, 19.1963, 18.0572])

dfreqs_c2 = (np.asarray([1, 1, 1, 2]) * (freqs_c2b - freqs_c2)) / (2*np.sqrt(2*np.log(2)))

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
b3_avg = np.mean([467, 463, 464])
db3_avg = np.std([467, 463, 464])

freqs_c3 = np.asarray([19.8561, 19.8560, 19.8564, 18.6807])
freqs_c3b = np.asarray([19.8674, 19.8671, 19.8662, 18.6907])

dfreqs_c3 = (np.asarray([0.5, 1, 0.5, 2]) * (freqs_c3b - freqs_c3)) / (2*np.sqrt(2*np.log(2)))

freqs = np.vstack((freqs_c1, freqs_c2, freqs_c3)) * 1e6
dfreqs = np.vstack((dfreqs_c1, dfreqs_c2, dfreqs_c3)) * 1e6
bfields = np.asarray([b1_avg, b2_avg, b3_avg]) * 1e-3
dbfields = np.asarray([db1_avg, db2_avg, db3_avg]) * 1e-3
dbfields[0] = dbfields[1] = 1e-3

gwfit, gwpopt, gwperr, gwchisq, gwchire = linearLSF(bfields, freqs[:,0], dfreqs[:,0])
ggfit, ggpopt, ggperr, ggchisq, ggchire = linearLSF(bfields, freqs[:,1], dfreqs[:,1])
gpfit, gppopt, gpperr, gpchisq, gpchire = linearLSF(bfields, freqs[:,2], dfreqs[:,2])
gtfit, gtpopt, gtperr, gtchisq, gtchire = linearLSF(bfields, freqs[:,3], dfreqs[:,3])

gw = ufloat(gwpopt[1]*h/mu, gwperr[1]*h/mu)
gg = ufloat(ggpopt[1]*h/mu, ggperr[1]*h/mu)
gp = ufloat(gppopt[1]*h/mu, gpperr[1]*h/mu)
gt = ufloat(gtpopt[1]*h/mu, gtperr[1]*h/mu)

fig, ax = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True)
fig.suptitle('Hydrogen g-factor', fontweight = 'bold')
ax[0].errorbar(bfields, freqs[:,0], xerr = dbfields, yerr = dfreqs[:,0], fmt='.', label = 'Data')
ax[0].plot(bfields, gwfit, label = 'Fit')
ax[0].legend()
ax[0].grid()
ax[0].set_ylabel('Frequency (Hz)')
ax[0].set_title('Water')

ax[1].errorbar(bfields, freqs[:,1], xerr = dbfields, yerr = dfreqs[:,1], fmt = '.', label = 'Data')
ax[1].plot(bfields, ggfit, label = 'Fit')
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('Magnetic Field (T)')
ax[1].set_title('Glycerine')

ax[2].errorbar(bfields, freqs[:,2], xerr = dbfields, yerr = dfreqs[:,2], fmt = '.', label = 'Data')
ax[2].plot(bfields, gpfit, label = 'Fit')
ax[2].legend()
ax[2].grid()
ax[2].set_title('Polystyrene')
plt.ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)

fig2, ax2 = plt.subplots()
fig2.suptitle('Hydrogen g-factor', fontweight = 'bold')
ax2.set_title('Overplot')
ax2.errorbar(bfields, freqs[:,0], xerr = dbfields, yerr = dfreqs[:,0], fmt = '.', label = 'Water Data')
ax2.plot(bfields, gwfit, label = 'Water Fit')
ax2.errorbar(bfields, freqs[:,1], xerr = dbfields, yerr = dfreqs[:,1], fmt = '.', label = 'Glycerine Data')
ax2.plot(bfields, ggfit, label = 'Glycerine Fit')
ax2.errorbar(bfields, freqs[:,2], xerr = dbfields, yerr = dfreqs[:,2], fmt = '.', label = 'Polystyrene Data')
ax2.plot(bfields, gpfit, label = 'Polystyrene Fit')
ax2.legend()
ax2.grid()
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Magnetic Field (T)')
plt.ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)

fig3, ax3 = plt.subplots()
fig3.suptitle('Fluorine g-factor', fontweight = 'bold')
ax3.set_title('Teflon')
ax3.errorbar(bfields, freqs[:,3], xerr = dbfields, yerr = dfreqs[:,3], fmt = '.', label = 'Data')
ax3.plot(bfields, gtfit, label = 'Fit')
ax3.legend()
ax3.grid()
ax3.set_ylabel('Frequency (Hz)')
ax3.set_xlabel('Magnetic Field (T)')
plt.ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)