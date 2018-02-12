import numpy as np
import matplotlib.pyplot as plt

def func3(x):
    f = x**3 - x
    return f

def func4(x):
    f = x**2 - 4*x*np.sin(x) + 4*(np.sin(x))**2
    return f

def hermite(x,n):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        h = 2 * x * hermite(x,n-1) - 2 * (n-1) * hermite(x,n-2)
        return h

def bisection(function, a, b):
    fa = function(a)
    fb = function(b)
    e = 10**(-5)
    d = e
    if (fa*fb > 0):
        print('Root not within interval')
        return
    while (np.abs(a - b) >= e):
        c = (a+b)/2
        fc = function(c)
        if (fc == 0) or (np.abs(fc) < d):
            break
        if (fa*fc < 0):
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c

def newtonraphson():
    
    

