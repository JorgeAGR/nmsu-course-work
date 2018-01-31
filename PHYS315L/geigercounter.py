import numpy
import matplotlib.pyplot as plt
import openpyxl

def xlsColumnData(xlssheet, min, max, column):
    array = []
    for i in range(min,max):
        d = xlssheet.cell(row = i, column = column).value
        array.append(d)
    return array

dataWB = openpyxl.load_workbook('Geigercounter.xlsx')

dataSheet = dataWB.get_sheet_by_name('Geigercounter')

controlData = xlsColumnData(dataSheet,1,101,2)

measurementData = xlsColumnData(dataSheet,1,101,4)

sampleSize = range(1,len(measurementData)+1)

f, ax = plt.subplots(1, 2)

#Displays graphs of experimental data (sample vs counts)
ax[0].plot(sampleSize, controlData, 'o')
ax[0].set_xlim(0,100)
ax[0].set_ylim(0,6)
ax[0].set_xticks(range(0,100,10))
ax[0].grid()

ax[1].plot(sampleSize, measurementData, 'o')
ax[1].set_xlim(0,100)
ax[1].set_ylim(0,100)
ax[1].set_yticks(range(0,110,10))
ax[1].set_xticks(range(0,100,10))
ax[1].grid()