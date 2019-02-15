#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:17:48 2018

@author: jorgeagr
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def golden_ratio(x):
    return x / ( (np.sqrt(5) + 1) / 2 )


class LSF(object):
    
    def __init__(self, x, y, sx, sy, fit='linear', n=1):
        
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.sx = np.asarray(sx)
        self.sy = np.asarray(sy)
        self.fit = fit
        self.n = n # Should be 1 unless dealing with polys
        
        self.fit_poly()
        self.evaluate()
        
        print(self.a, self.da, self.rechisq)
    
    def fit_poly(self):
        # Data Vector (Outputs)
        X = np.zeros(shape = (self.n+1, 1))
        # Measurement Matrix (Inputs)
        M = np.zeros(shape = (self.n+1, self.n+1))
        # Coeffficients Vector
        self.a = np.zeros(self.n+1)
        counter = 0
        while True:
            counter += 1
            #dydx = np.zeros(len(x))
            A = self.a.copy()
            self.w = 1 / (self.sy ** 2 + (self.dfunc(self.x) * self.sx)**2)
            for i in range(self.n+1):
                for j in range(self.n+1):
                    M[i][j] = np.sum(self.w * self.x**(j+i))
                X[i][0] = np.sum(self.w * self.y * self.x ** i)
            self.a = np.dot(np.linalg.inv(M), X).flatten()
            if (np.abs(A - self.a).all() < 1e-12) or (counter == 100):
                break
        
        self.a = self.a.flatten()
        self.da = np.sqrt(np.linalg.inv(M).diagonal())
        
    def evaluate(self):
        chisq = np.sum((self.y - self.func(self.x))**2 * self.w)
        self.rechisq = chisq / (len(self.x) - 2)
        
    def func(self, x):
        f = 0
        for i in range(len(self.a)):
            f += self.a[i] * x ** (i)
        return f
    
    def dfunc(self, x):
        f = 0
        for i in range(len(self.a)):
            f += self.a[i] * i * x ** (i-1)
        return f

# --------------------------------------------------------------------------- #    

class Canvas(tk.Frame):
    
    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        
        self.f, self.ax = plt.subplots(figsize = (9,golden_ratio(5)))
        self.ax.grid()
        
        self.x = []
        self.y = []
        
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

class DataRow(tk.Frame):
    
    def __init__(self, parent, row):
        
        tk.Frame.__init__(self, parent)
    
        self.x = tk.DoubleVar()
        self.x.set('')
        self.xcell = tk.Entry(parent, width = 10, textvariable = self.x)
        self.xcell.grid(row = row, column = 0)
        
        self.sx = tk.DoubleVar()
        self.sx.set('')
        self.sxcell = tk.Entry(parent, width = 10, textvariable = self.sx)
        self.sxcell.grid(row = row, column = 1)
        
        self.y = tk.DoubleVar()
        self.y.set('')
        self.ycell = tk.Entry(parent, width = 10, textvariable = self.y)
        self.ycell.grid(row = row, column = 2)
        
        self.sy = tk.DoubleVar()
        self.sy.set('')
        self.sycell = tk.Entry(parent, width = 10, textvariable = self.sy)
        self.sycell.grid(row = row, column = 3)
        
        self.xcell.bind("<FocusOut>", self.return_x)
        self.ycell.bind("<FocusOut>", self.return_y)
        
    def return_x(self, event):
        print(self.x.get())
    
    def return_y(self, event):
        print(self.y.get())

class Spreadsheet(tk.Frame):
    
    def __init__(self, parent, cells=20):
    
        tk.Frame.__init__(self, parent)
        
        self.data = pd.DataFrame(columns = ['x', 'sx', 'y', 'sy'])
        
        self.sheet = {}
        
        self.xlabel = tk.Label(self, text = 'x')
        self.xlabel.grid(row = 0, column = 0)
        
        self.sxlabel = tk.Label(self, text = 'sx')
        self.sxlabel.grid(row = 0, column = 1)
        
        self.ylabel = tk.Label(self, text = 'y')
        self.ylabel.grid(row = 0, column = 2)
        
        self.sylabel = tk.Label(self, text = 'sy')
        self.sylabel.grid(row = 0, column = 3)
        
        for i in range(cells):
            self.add_Row()
    
    def add_Row(self):
        i = len(self.sheet.keys())
        self.sheet[i] = DataRow(self, i+1)
        self.data.loc[i] = np.zeros(4)
    
    def cell_To_Data(self):
        for i in range(len(self.data)):
            self.data.iloc[i] = [self.sheet[i].x.get(), self.sheet[i].y.get()]
        print(self.data)


class MainApp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        
        tk.Tk.wm_title(self, 'Least Squares Fitting App')
        tk.Tk.wm_geometry(self, '900x500')
        
        self.SpreadSheet = Spreadsheet(self)
        self.SpreadSheet.grid(row = 0, column = 0, sticky = 'NS', padx = 5, pady = 5)
        
        rightFrame = tk.Frame(self)
        rightFrame.grid(row = 0, column = 1, sticky = 'NWSE')
        
        toolsMenu = tk.Frame(rightFrame)
        toolsMenu.grid(row = 0, column = 1, sticky = 'NSWE')
        
        self.TestButton = tk.Button(toolsMenu, text = 'Add Row', command = self.SpreadSheet.add_Row)
        self.TestButton.grid(row = 0, column = 0)
        
        self.YButton = tk.Button(toolsMenu, text = 'Y Error', command = None)
        self.YButton.grid(row = 0, column = 1)
        
        self.XYButton = tk.Button(toolsMenu, text = 'X & Y Errors', command = None)
        self.XYButton.grid(row = 0, column = 2)
        
        self.PlotCanvas = Canvas(rightFrame)
        self.PlotCanvas.grid(row = 1, column = 1, sticky = 'NSWE')
        
        self.VarianceCanvas = Canvas(rightFrame)
        self.VarianceCanvas.grid(row = 2, column = 1, sticky = 'NSWE')
        
app = MainApp()
#app.tk.call('tk', 'scaling', 2.0)
#app.protocol("WM_DELETE_WINDOW", exiting)
app.mainloop()