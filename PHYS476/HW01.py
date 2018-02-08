import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# == 1 - Machine Numbers == # 

'''def binary(number):
    def binexp(x):
        y = 1/(2**(x)
        return y
    n = 0
    fracc = binexp(n)
    while fracc > number:
        n += 1
        fracc = binexp(n)
    else:'''
        
# 3 - Hermite Polynomials (Explicit and Recursion)

x3 = np.linspace(-3.0, 3.0, num = 100)
nMaxH = 5 #Number of Hermite polynomials desired

X = sym.Symbol('x')
H = [] 
Hf = []

for n in range(nMaxH):
    h = sym.diff(sym.exp(-X**2), X, n) * ((-1)**n) * (sym.exp(X**2))
    H.append(h)
    Hf.append(sym.lambdify(X, H[n], 'numpy'))

def hermite(x,n):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        h = 2 * x * hermite(x,n-1) - 2 * (n-1) * hermite(x,n-2)
        return h

hRecursion = []
hExplicit = []
for n in range(nMaxH):
    hR = list(map(lambda x: hermite(x,n), x3))
    hE = list(map(lambda x: Hf[n](x), x3))
    hRecursion.append(hR)
    hExplicit.append(hE)

figHR, axHR = plt.subplots()
for n in range(nMaxH):
    plt.plot(x3, hRecursion[n])
axHR.set_ylim(-25, 25)
axHR.set_xlim(-3,3)
axHR.grid()
axHR.set_title('Recursion H$_n$')

figHE, axHE = plt.subplots()
for n in range(nMaxH):
    plt.plot(x3, hExplicit[n], '--')
axHE.set_ylim(-25, 25)
axHE.set_xlim(-3,3)
axHE.grid()
axHE.set_title('Explicit H$_n$')

figH3, axH3 = plt.subplots()
axH3.plot(x3, hRecursion[3], 'c')
axH3.plot(x3, hExplicit[3], 'b:')
axH3.set_ylim(-25,25)
axH3.set_xlim(-3,3)
axH3.grid()
axH3.set_title('H$_3$')

figH4, axH4 = plt.subplots()
axH4.plot(x3, hRecursion[4], 'c')
axH4.plot(x3, hExplicit[4], 'b:')
axH4.set_ylim(-25,25)
axH4.set_xlim(-3,3)
axH4.grid()
axH4.set_title('H$_4$')

# 4 - Legendre Polynomials

def legendre(x,n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        p = ((2*(n-1) + 1) * x * legendre(x,n-1) - (n-1) * legendre(x,n-2)) / (n)
        return p

x4 = np.linspace(-1, 1, num = 100)
nMaxP = 6

pRecursion = []
for n in range(nMaxP):
    p = list(map(lambda x: legendre(x,n), x4))
    pRecursion.append(p)

figP, axP = plt.subplots()
for n in range(nMaxP):
    axP.plot(x4, pRecursion[n])
axP.set_ylim(-1,1.05)
axP.set_xlim(-1,1)
axP.grid()
axP.set_title('Legendre Polynomials')

# 5 - Compute numerical derivatives

x5 = 10

f = x5**4

h = np.linspace(1, 0, num = 250, endpoint = False) 
for i in h:
    df2p = ((x5+h)**4 - (x5-h)**4)/(2*h)
    df4p = ((x5-2*h)**4 - 8*(x5-h)**4 + 8*(x5+h)**4 - (x5+2*h)**4)/(12*h)