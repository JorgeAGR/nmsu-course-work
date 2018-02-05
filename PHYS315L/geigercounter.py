import numpy as np
import math
import matplotlib.pyplot as plt
import openpyxl
from scipy.optimize import curve_fit

# == Defining functions required == #

def function(x, a, b):
    return b * x + a

def xlsColumnData(xlssheet, min, max, column):
    array = []
    for i in range(min,max):
        d = xlssheet.cell(row = i, column = column).value
        array.append(d)
    return array

def poisson(array):
    arrayElements = list(range(max(array)+1))
    
    arrayMean = np.mean(array)
    
    probabilityElements = dict.fromkeys(arrayElements,0)
    for i in arrayElements:
        probability = ((arrayMean ** i) * (np.exp( - arrayMean))) / (math.factorial( i ))
        probabilityElements[i] = probability
    
    arrayPoison = []
    for i in arrayElements:
        success = len(array) * probabilityElements[i]
        arrayPoison.append(success)
    
    return arrayPoison

def normal(array):
    arrayElements = list(range(max(array)+1))
    
    arrayMean = np.mean(array)
    
    probabilityElements = dict.fromkeys(arrayElements,0)
    for i in arrayElements:
        probability = (np.exp((-(i - arrayMean)**2) / (2*arrayMean))) / (np.sqrt(2*np.pi*arrayMean)) 
        probabilityElements[i] = probability
    
    arrayNormal = []
    for i in arrayElements:
        success = probabilityElements[i] * 500
        arrayNormal.append(success)
    
    return arrayNormal

# == 2 - Curve Fitting == #

freq = np.array([5.19, 5.49, 6.88, 7.41, 8.20])
volt = np.array([0.702, 0.810, 1.378, 1.596, 1.894])
dvolt = np.array([0.010, 0.010, 0.010, 0.010, 0.010])

popt, pcov = curve_fit(function, freq, volt)

fit = function(freq, *popt)

f2, ax2 = plt.subplots()
#ax2 = plt.plot(freq, volt, '.b')
plt.errorbar(freq, volt, yerr = dvolt, fmt = '.', mfc = 'b')
ax2 = plt.plot(freq, fit, 'c')

chisq = np.sum(((volt - fit)/dvolt)**2)
chire = chisq / (len(freq) - len(popt))

# Test a
#a_abv = np.linspace(popt[0]-0.5,popt[0],endpoint = False)
#a_bel = np.linspace(popt[0],popt[0]+0.5)
a_var = np.hstack((np.linspace(popt[0]-0.5,popt[0],num=10,endpoint = False), np.linspace(popt[0],popt[0]+0.5,num=10)))
chisq_da = []
for a in a_var:
    fit_a = function(freq, a, popt[1])
    chi = np.sum((volt - fit_a)/dvolt)**2
    chisq_da.append(chi)

# Turn all of this above into functional form
# How the fuck can I determine uncertainties here...

# == 3 - Low and High Rate Count Data Analysis == #
    
dataWB = openpyxl.load_workbook('Geigercounter.xlsx')

dataSheet = dataWB.get_sheet_by_name('Geigercounter')

controlData = xlsColumnData(dataSheet,1,101,2)

measurementData = xlsColumnData(dataSheet,1,101,4)

del controlData[0]
del measurementData[0]

sampleSize = range(1,len(measurementData)+1)

# Displays graphs of experimental data (sample vs counts)
f3a, ax3a = plt.subplots(1, 2)
ax3a[0].plot(sampleSize, controlData, 'o')
ax3a[0].set_xlim(0,100)
ax3a[0].set_ylim(0,6)
ax3a[0].set_xticks(range(0,100,10))
ax3a[0].grid()

ax3a[1].plot(sampleSize, measurementData, 'o')
ax3a[1].set_xlim(0,100)
ax3a[1].set_ylim(0,100)
ax3a[1].set_yticks(range(0,110,10))
ax3a[1].set_xticks(range(0,100,10))
ax3a[1].grid()

controlPoisson = poisson(controlData)
measurementNormal = normal(measurementData)

f3b, ax3b = plt.subplots(1,2)
ax3b[0].set_title('Control')
ax3b[0].hist(controlData)
ax3b[0].plot(controlPoisson)
ax3b[0].set_xlim(min(controlData),max(controlData))
ax3b[0].set_xticks(range(0,max(controlData)+6))
ax3b[1].set_title('Measurements')
ax3b[1].hist(measurementData)
ax3b[1].plot(measurementNormal)
ax3b[1].set_xticks(range(0,100,10))