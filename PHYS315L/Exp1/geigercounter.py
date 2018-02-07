import numpy as np
import math
from fractions import Fraction
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
    
    return arrayPoison, probabilityElements

def normal(array):
    arrayElements = list(range(max(array)+1))
    
    arrayMean = np.mean(array)
    
    probabilityElements = dict.fromkeys(arrayElements,0)
    for i in arrayElements:
        probability = (np.exp((-(i - arrayMean)**2) / (2*arrayMean))) / (np.sqrt(2*np.pi*arrayMean)) 
        probabilityElements[i] = probability
    
    arrayNormal = []
    for i in arrayElements:
        success = probabilityElements[i] * len(array)
        arrayNormal.append(success)
    
    return arrayNormal, arrayElements

def linearLSF(x, y, dy):
    def linear(x, a, b):
        return b * x + a
    
    popt, pcov = curve_fit(linear, x, y)
    
    fit = linear(x, *popt)
    
    chisq = np.sum(((y - fit)/dy)**2) #Chi Squared
    chire = chisq / (len(x) - len(popt)) # Chi-Reduced (Chi Squared / Dof)
    perr = np.sqrt(np.diag(pcov)) # Error in parameters a, b
    
    return fit, popt, perr, chisq, chire

def testDeviation(array, std):
    mean = np.mean(array)
    counter = 0
    for i in array:
        d = np.abs(i - mean)
        if d > std:
            counter += 1
    return counter/len(array) 

# == 2 - Curve Fitting == #

freq = np.array([5.19, 5.49, 6.88, 7.41, 8.20])
volt = np.array([0.702, 0.810, 1.378, 1.596, 1.894])
dvolt = np.array([0.010, 0.010, 0.010, 0.010, 0.010])

fit, popt, perr, chisq, chire = linearLSF(freq, volt, dvolt)

f2, ax2 = plt.subplots()
f2.suptitle('Least Squares Fitting', fontweight = 'bold')
ax2.errorbar(freq, volt, yerr = dvolt, fmt = '.', mfc = 'b', label = 'Data')
ax2.plot(freq, fit, 'c', label = '$\chi^2$ Function')
ax2.set_xlabel('Frequency (x$10^{14}$ 5Hz)')
ax2.set_ylabel('Voltage (V)')
ax2.grid()
text_num2 = 'Parameters: \na = %.3f \xb1 %.3f\nb = %.3f \xb1 %.3f\n\n $\chi^2/\\nu$ = %.3f' % (popt[0], perr[0], popt[1], perr[1], chire) # \xb1 gives plus-minus sign!
ax2.text(7,1, text_num2, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax2.legend(fontsize = 'medium')

# == 3 - Low and High Rate Count Data Analysis == #
    
dataWB = openpyxl.load_workbook('Geigercounter.xlsx')

dataSheet = dataWB.get_sheet_by_name('Geigercounter')

controlData = xlsColumnData(dataSheet,1,101,2)

measurementData = xlsColumnData(dataSheet,1,101,4)

del controlData[0]
del measurementData[0]

sampleSizeC = range(1, len(controlData)+1)
sampleSizeM = range(1, len(measurementData)+1)

controlMean = np.mean(controlData)
controlMeanSqrt = np.sqrt(controlMean)
controlSTD = np.std(controlData)
controlSTDSq = controlSTD**2

measurementMean = np.mean(measurementData)
measurementMeanSqrt = np.sqrt(measurementMean)
measurementSTD = np.std(measurementData)
measurementSTDSq = measurementSTD**2

controlPoisson, controlSuccess = poisson(controlData)
controlNormal, controlBins = normal(controlData)#, 99)
measurementPoisson, measurementSuccess = poisson(measurementData)
measurementNormal, measurementBins = normal(measurementData)#, 500)

controlFit, cPopt, cPerr, cChiSq, cChiRe = linearLSF(sampleSizeC, controlData, controlMeanSqrt)
measurementFit, mPopt, mPerr, mChiSq, mChiRe = linearLSF(sampleSizeM, measurementData, measurementMeanSqrt)

# Displays graphs of experimental data (sample vs counts)
f3a, ax3a = plt.subplots(1, 2)
f3a.suptitle('Radioactive Decay: Counts', fontweight = 'bold')

ax3a[0].set_title('Control')
ax3a[0].plot(sampleSizeC, controlData, 'o', label = 'Data')
ax3a[0].plot(sampleSizeC, controlFit, label = 'Linear Fit')
ax3a[0].set_xlim(0,100)
ax3a[0].set_ylim(0,6)
ax3a[0].set_xticks(range(0,100,10))
ax3a[0].set_xlabel('Sample #')
ax3a[0].set_ylabel('Counts')
ax3a[0].grid()
text_num3a = 'Parameters: \na = %.3f \xb1 %.3f\nb = %.3f \xb1 %.3f\n\n $\chi^2/\\nu$ = %.3f' % (cPopt[0], cPerr[0], cPopt[1], cPerr[1], cChiRe) # \xb1 gives plus-minus sign!
ax3a[0].text(60,5, text_num3a, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax3a[0].legend(fontsize = 'medium')

ax3a[1].set_title('Measurements')
ax3a[1].plot(sampleSizeM, measurementData, 'o', label = 'Data')
ax3a[1].plot(sampleSizeM, measurementFit, label = 'Linear Fit')
ax3a[1].set_xlim(0,100)
ax3a[1].set_ylim(0,100)
ax3a[1].set_yticks(range(0,110,10))
ax3a[1].set_xticks(range(0,100,10))
ax3a[1].set_xlabel('Sample #')
ax3a[1].set_ylabel('Counts')
ax3a[1].grid()
text_num3b = 'Parameters: \na = %.3f \xb1 %.3f\nb = %.3f \xb1 %.3f\n\n $\chi^2/\\nu$ = %.3f' % (mPopt[0], mPerr[0], mPopt[1], mPerr[1], mChiRe) # \xb1 gives plus-minus sign!
ax3a[1].text(60,5, text_num3b, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax3a[1].legend(fontsize = 'medium')

# Histograms and Curve Fits
f3b, ax3b = plt.subplots(1,2)
f3b.suptitle('Radioactive Decay: Statistics', fontweight = 'bold')

ax3b[0].set_title('Control')
ax3b[0].hist(controlData, bins = controlBins, color = 'burlywood')
ax3b[0].plot(controlPoisson, label = 'Poisson Fit')
ax3b[0].plot(controlNormal, label = 'Normal Fit')
ax3b[0].set_xlim(min(controlData),max(controlData))
ax3b[0].set_xticks(range(0,max(controlData)+6))
ax3b[0].set_xlabel('Counts')
ax3b[0].set_ylabel('Frequency')
ax3b[0].legend(fontsize = 'medium')

ax3b[1].set_title('Measurements')
ax3b[1].hist(measurementData, bins = measurementBins, color = 'burlywood')
ax3b[1].plot(measurementPoisson, label = 'Poisson Fit')
ax3b[1].plot(measurementNormal, label = 'Normal Fit')
ax3b[1].set_xticks(range(0,100,10))
ax3b[1].set_xlabel('Counts')
ax3b[1].set_ylabel('Frequency')
ax3b[1].legend(fontsize = 'medium')


plt.show()