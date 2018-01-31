import numpy as np
import math
import matplotlib.pyplot as plt
import openpyxl

def xlsColumnData(xlssheet, min, max, column):
    array = []
    for i in range(min,max):
        d = xlssheet.cell(row = i, column = column).value
        array.append(d)
    return array

def frequencyData(array):
    frequency = []
    for n in range(max(array)+1):
        counter = 0
        for i in array:
            if i == n:
                counter += 1
        frequency.append(counter)
    return frequency

def setOfElements(array):
    elements = []
    for i in array:
        if i not in elements:
            elements.append(i)
    elements.sort()
    elements = tuple(elements)
    return elements

def poisson(array):
    arrayElements = []
    for i in array:
        if i not in arrayElements:
            arrayElements.append(i)
    arrayElements.sort()
    arrayElements = tuple(arrayElements)
    
    arrayMean = np.mean(array)
    
    probabilityElements = dict.fromkeys(arrayElements,0)
    for i in arrayElements:
        probability = ((arrayMean ** i) * (np.exp( - controlMean))) / (math.factorial( i ))
        probabilityElements[i] = probability
    
    arrayPoison = []
    for i in arrayElements:
        success = len(array) * probabilityElements[i]
        arrayPoison.append(success)
    
    return arrayPoison

dataWB = openpyxl.load_workbook('Geigercounter.xlsx')

dataSheet = dataWB.get_sheet_by_name('Geigercounter')

controlData = xlsColumnData(dataSheet,1,101,2)

measurementData = xlsColumnData(dataSheet,1,101,4)

del controlData[0]
del measurementData[0]

sampleSize = range(1,len(measurementData)+1)

#Displays graphs of experimental data (sample vs counts)
'''f1, ax1 = plt.subplots(1, 2)
ax1[0].plot(sampleSize, controlData, 'o')
ax1[0].set_xlim(0,100)
ax1[0].set_ylim(0,6)
ax1[0].set_xticks(range(0,100,10))
ax1[0].grid()

ax1[1].plot(sampleSize, measurementData, 'o')
ax1[1].set_xlim(0,100)
ax1[1].set_ylim(0,100)
ax1[1].set_yticks(range(0,110,10))
ax1[1].set_xticks(range(0,100,10))
ax1[1].grid()'''        

controlFrequency = frequencyData(controlData)
controlPoisson = poisson(controlData)

f2, ax2 = plt.subplots(1,2)
ax2[0].set_title('Control')
ax2[0].plot(controlFrequency,'o')
ax2[0].plot(controlPoisson)
ax2[0].set_xlim(min(controlData),max(controlData))
ax2[0].set_xticks(range(0,max(controlData)+6))
ax2[1].set_title('Measurements')
ax2[1].hist(measurementData)
ax2[1].set_xticks(range(0,100,10))