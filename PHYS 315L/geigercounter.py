import numpy
import matplotlib.pyplot as plt
import openpyxl

def xlsColumnData(xlssheet, min, max, column):
    array = []
    for i in range(min,max):
        d = xlssheet.cell(row = i, column = column).value
        array.append(d)
    return array

datawb = openpyxl.load_workbook('Geigercounter.xlsx')

datasheet = datawb.get_sheet_by_name('Geigercounter')

control = xlsColumnData(datasheet,1,101,2)

measurement = xlsColumnData(datasheet,1,101,4)

f = 