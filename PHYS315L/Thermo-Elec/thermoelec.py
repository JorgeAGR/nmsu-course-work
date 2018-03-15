import numpy as np
import csv
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

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

sm_background_cols = 31
sm_heatingup_cols = 81
sm_coolingdown_cols = 339

data = []
with open('thermoresist.csv', 'r') as file:
    reader = csv.reader(file, delimiter = ',')  
    for i,x in enumerate(reader):
        if i not in (0,1):
            data.append(x)

for row in range(len(data)):
    for element in range(len(data[row])):
        try:
            data[row][element] = float(data[row][element])
        except:
            None

sm_test_time = []
sm_test_v = []
sm_test_temp = []
for i in data[0:31]:
    sm_test_time.append(i[0])
    sm_test_v.append(i[1])
    sm_test_temp.append(i[2])

sm_heat_time = []
sm_heat_v = []
sm_heat_temp = []
for i in data[0:81]:
    sm_heat_time.append(i[3])
    sm_heat_v.append(i[4])
    sm_heat_temp.append(i[5])

sm_cool_time = []
sm_cool_v = []
sm_cool_temp = []
for i in data[0:339]:
    sm_cool_time.append(i[6])   
    sm_cool_v.append(i[7])
    sm_cool_temp.append(i[8])

met_test_time = []
met_test_v = []
met_test_temp = []
for i in data[0:31]:
    met_test_time.append(i[9])
    met_test_v.append(i[10])
    met_test_temp.append(i[11])

met_heat_time = []
met_heat_v = []
met_heat_temp = []
for i in data[0:63]:
    met_heat_time.append(i[12])
    met_heat_v.append(i[13])
    met_heat_temp.append(i[14])

met_cool_time = []
met_cool_v = []
met_cool_temp = []
for i in data[0:349]:
    met_cool_time.append(i[15])
    met_cool_v.append(i[16])
    met_cool_temp.append(i[17])

sm_current = ufloat(0.100, 0.001)
met_current = 0.05

sm_temp_std = round(np.std(sm_test_temp), 1)
sm_v_std = round(np.std(sm_test_v), 3)

met_temp_std = round(np.std(met_test_temp), 1) # No noise here! So error is instrumental
met_temp_std = 0.1
met_v_std = round(np.std(met_test_v), 3)

'''
# Data of semiconductor (0.1 A used)
figsmdata, axsmdata = plt.subplots()
axsmdata.errorbar(sm_heat_temp, sm_heat_v, xerr = sm_temp_std, yerr = sm_v_std, fmt = '.', label = 'Heating Up')
axsmdata.errorbar(sm_cool_temp, sm_cool_v, xerr = sm_temp_std, yerr = sm_v_std, fmt = '.', label = 'Cooling Down')
#axsmdata.set_xlim(0,None)
#axsmdata.set_ylim(0,None)
axsmdata.set_title('NTC Resistor (Semiconductor)')
axsmdata.set_xlabel('Temperature (C)')
axsmdata.set_ylabel('Voltage (V)')
axsmdata.grid()
axsmdata.legend()
'''
'''
# Data of metal (0.05 A used)
figmetdata, axmetdata = plt.subplots()
axmetdata.errorbar(met_heat_temp[4:-5], met_heat_v[4:-5], xerr = met_temp_std, yerr = met_v_std, fmt = '.', label = 'Heating Up')
axmetdata.errorbar(met_cool_temp, met_cool_v, xerr = met_temp_std, yerr = met_v_std, fmt = '.', label = 'Cooling Down')
#axmetdata.set_xlim(0,None)
#axmetdata.set_ylim(0,None)
axmetdata.set_title('Platinum Wire (Metal)')
axmetdata.set_xlabel('Temperature (C)')
axmetdata.set_ylabel('Voltage (V)')
axmetdata.grid()
axmetdata.legend()
'''

# Semiconductor - Fit of both combined --- Check if its temp or volt thats causing ttrouble...
sm_temp = np.hstack((sm_heat_temp[4:-5], sm_cool_temp[50:]))
sm_v = np.hstack((sm_heat_v[4:-5], sm_cool_v[50:]))
sm_fit, sm_popt, sm_perr, sm_chisq, sm_chire = linearLSF(sm_temp, sm_v, sm_v_std)
'''
figsmtot, axsmtot = plt.subplots()
axsmtot.plot(sm_temp, sm_fit)
axsmtot.errorbar(sm_temp, sm_v, xerr = sm_temp_std, yerr = sm_v_std, fmt = '.')
axsmtot.set_title(')
textsm = 'Parameters: \n$(R/n_f)^2$ = %.1f \xb1 %.1f\nR = %.1f \xb1 %.1f $m^{-1}$\n\n $\chi^2/\\nu$ = %.3e' % (rPopt[0], rPerr[0], rPopt[1], rPerr[1], rChiRe)
axH.text(0.05,1600000, textH, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})s
'''

# Metal - Fit of each heating and cooling
met_heat_fit, met_heat_popt, met_heat_perr, met_heat_chisq, met_heat_chire = linearLSF(np.asarray(met_heat_temp[3:-1]),
                                                                                       np.asarray(met_heat_v[3:-1]),
                                                                                       met_v_std)
met_cool_fit, met_cool_popt, met_cool_perr, met_cool_chisq, met_cool_chire = linearLSF(np.asarray(met_cool_temp[64:]), 
                                                                                       np.asarray(met_cool_v[64:]),
                                                                                       met_v_std)

met_heat_r0 = ufloat(met_heat_popt[0], met_heat_perr[0])
met_heat_ar0 = ufloat(met_heat_popt[1], met_heat_perr[1])
met_heat_a = met_heat_ar0 / met_heat_r0
met_cool_r0 = ufloat(met_cool_popt[0], met_cool_perr[0])
met_cool_ar0 = ufloat(met_cool_popt[1], met_cool_perr[1])
met_cool_a = met_cool_ar0 / met_cool_r0

'''
figmet, axmet = plt.subplots(ncols = 2, sharex = True, sharey = True)
figmet.suptitle('Metal', fontsize = 18)
axmet[0].plot(met_heat_temp[3:-1], met_heat_fit, label = 'Fit')
axmet[0].errorbar(met_heat_temp[3:-1], met_heat_v[3:-1], xerr = met_temp_std, yerr = met_v_std, fmt = '.', label = 'Data')
axmet[1].plot(met_cool_temp[64:], met_cool_fit, label = 'Fit')
axmet[1].errorbar(met_cool_temp[64:], met_cool_v[64:], xerr = met_temp_std, yerr = met_v_std, fmt = '.', label = 'Data')
axmet[0].set_title('Heating Up')
axmet[1].set_title('Cooling Down')
axmet[0].set_xlabel('Temperature (C)')
axmet[1].set_xlabel('Temperature (C)')
axmet[0].set_ylabel('Voltage (V)')
axmet[0].grid() 
axmet[1].grid()
axmet[0].legend()
axmet[1].legend()
'''

#textmetheat = 'Parameters: \n$R_0$ = %.3e \xb1 %.3e\n$alpha R_0$ = %.3e \xb1 %.3e \n\n $\chi^2 / \nu$ = %.3e' % (met_heat_popt[0], met_heat_perr[0], met_heat_popt[1], met_heat_perr[1], met_heat_chire)
#axmet[0].text(100, 6.0, textmetheat, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})

# Semiconductor - 
sm_heat_res = unumpy.uarray(sm_heat_v[4:-5], std_devs = sm_v_std) / sm_current
sm_heat_res = unumpy.uarray(np.round(unumpy.nominal_values(sm_heat_res), decimals = 3),
                            np.round(unumpy.std_devs(sm_heat_res), decimals = 3))
sm_heat_lnr = unumpy.uarray(np.log(unumpy.nominal_values(sm_heat_res)),
                                   unumpy.std_devs(sm_heat_res) / unumpy.nominal_values(sm_heat_res))

sm_heat_invt = 1 / unumpy.uarray(sm_heat_temp[4:-5], std_devs = sm_temp_std)
sm_heat_invt = unumpy.uarray(np.round(unumpy.nominal_values(sm_heat_invt), decimals = 5),
                             np.round(unumpy.std_devs(sm_heat_invt), decimals = 5))

sm_heat_fit, sm_heat_popt, sm_heat_perr, sm_heat_chisq, sm_heat_chire = linearLSF(unumpy.nominal_values(sm_heat_invt),
                                                                                  unumpy.nominal_values(sm_heat_lnr),
                                                                                  unumpy.std_devs(sm_heat_lnr))

sm_cool_res = unumpy.uarray(sm_cool_v[50:], std_devs = sm_v_std) / sm_current
sm_cool_res = unumpy.uarray(np.round(unumpy.nominal_values(sm_cool_res), decimals = 3),
                            np.round(unumpy.std_devs(sm_cool_res), decimals = 3))
sm_cool_lnr = unumpy.uarray(np.log(unumpy.nominal_values(sm_cool_res)),
                                   unumpy.std_devs(sm_cool_res) / unumpy.nominal_values(sm_cool_res))

sm_cool_invt = 1 / unumpy.uarray(sm_cool_temp[50:], std_devs = sm_temp_std)
sm_cool_invt = unumpy.uarray(np.round(unumpy.nominal_values(sm_cool_invt), decimals = 5),
                             np.round(unumpy.std_devs(sm_cool_invt), decimals = 5))

sm_cool_fit, sm_cool_popt, sm_cool_perr, sm_cool_chisq, sm_cool_chire = linearLSF(unumpy.nominal_values(sm_cool_invt),
                                                                                  unumpy.nominal_values(sm_cool_lnr),
                                                                                  unumpy.std_devs(sm_cool_lnr))

sm_heat_dE2Kb = ufloat(sm_heat_popt[1], sm_heat_perr[1])
sm_cool_dE2Kb = ufloat(sm_cool_popt[1], sm_cool_perr[1])

kb = 1.380649e-23
sm_heat_dE = sm_heat_dE2Kb * (2*kb)
sm_cool_dE = sm_cool_dE2Kb * (2*kb)

'''
figsm, axsm = plt.subplots(ncols = 2, sharex = True, sharey = True)
figsm.suptitle('Semiconductor', fontsize = 18)https://texblog.org/2014/06/24/big-o-and-related-notations-in-latex/?share=facebook&nb=1
axsm[0].plot(unumpy.nominal_values(sm_heat_invt), sm_heat_fit, label = 'Fit')
axsm[0].errorbar(unumpy.nominal_values(sm_heat_invt), unumpy.nominal_values(sm_heat_lnr), fmt = '.', label = 'Data')
axsm[1].plot(unumpy.nominal_values(sm_cool_invt), sm_cool_fit, label = 'Fit')
axsm[1].errorbar(unumpy.nominal_values(sm_cool_invt), unumpy.nominal_values(sm_cool_lnr), fmt = '.', label = 'Data')
axsm[0].set_title('Heating Up')
axsm[1].set_title('Cooling Down')
axsm[0].set_xlabel('1/T (1/C)')
axsm[1].set_xlabel('1/T (1/C)')
axsm[0].set_ylabel('ln(R)')
axsm[0].grid()
axsm[1].grid()
axsm[0].legend()
axsm[1].legend()
'''