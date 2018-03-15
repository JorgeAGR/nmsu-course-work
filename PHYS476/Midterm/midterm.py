import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as integrate
from scipy.interpolate import lagrange
from scipy.misc import derivative as df

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

def newtonraphson(func, dfunc, a, b, iterations):
    e = 10**(-5)
    d = e
    i = 0
    xn = (a + b) / 2
    while (i < iterations):
        if dfunc(xn) != 0:
            xn = xn - (func(xn) / dfunc(xn))
        else:
            print('Division by 0')
            return
        if (xn < a) or (xn > b):
            print('Out of bounds')
        elif (np.abs(func(xn)) < d):
            return xn
    print('Max iterations reached')

def function(x):
    f = x**3 - 5*x**2 + 8*x - 4
    return f

def dfunction(x):
    f = 3*x**2 - 10*x + 8
    return f

def func(x):
    f = 8*x**7 - 9*x**5 + 22*x - 7
    return f
'''
f1 = func(0)
f2 = func(2)
f3 = func(4)

def simpson(f, a, b, n):
    h=(b-a)/n
    k=0.0
    x=a + h
    for i in range(1,int(n/2) + 1):
        k += 4*f(x)
        x += 2*h

    x = a + 2*h
    for i in range(1,int(n/2)):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+k)

integral = []
for n in range(1,101):
    i = simpson(func, 0, 4, n)
    integral.append(i)

figint, axint = plt.subplots()
axint.loglog(range(1,101), np.abs(84 - np.asarray(integral)))
'''

def simpsons(f, a, b, N):
    h = (b - a) / N
    fa = f(a)
    fb = f(b)
    fint = fa + fb
    for n in range(1,N):
        if n%2 == 0:
            fn = 2*f(a + n*h)
            fint += fn
        else:
            fn = 4*f(a + n*h)
            fint += fn
    return (h/3)*fint

integral = []
error = []
for N in range(0, 1001):
    i = simpsons(func, 0, 100, 2*N+1)
    e = 9998500000109300 - i
    integral.append(i)
    error.append(e)

plt.plot(np.abs(error))
plt.xscale('log')
plt.yscale('log')
plt.grid()

x = np.linspace(0.8 , 2.5, num = 200)
y = function(x)
roots = np.array([newtonraphson(function, dfunction, 0, 1.5, 10), newtonraphson(function, dfunction, 1.5, 3, 10)])
froots = function(roots)
'''
figf, axf = plt.subplots()
axf.plot(x, y, label = r'$f(x)$')
axf.plot(roots, froots, '.', markersize = 12, label = r'$x = 1, 2$')
axf.set_title(r'$f(x) = x^3 - 5x^2 + 8x -4$')
axf.grid()
axf.legend()
'''