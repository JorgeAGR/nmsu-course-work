import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import matplotlib as mpl

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'

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

figHE, axHE = plt.subplots()
figHE.set_size_inches(9,9)
for n in range(nMaxH):
    text = 'H$_%i$' % n
    plt.plot(x3, hExplicit[n], '--', label = text)
axHE.set_ylim(-25, 25)
axHE.set_xlim(-3,3)
axHE.grid()
axHE.set_title('Explicit H$_n$')
axHE.legend()

figHR, axHR = plt.subplots()
figHR.set_size_inches(9,9)
for n in range(nMaxH):
    text = 'H$_%i$' % n
    plt.plot(x3, hRecursion[n], label = text)
axHR.set_ylim(-25, 25)
axHR.set_xlim(-3,3)
axHR.grid()
axHR.set_title('Recursion H$_n$')
axHR.legend()

figH3, axH3 = plt.subplots()
figH3.set_size_inches(9,9)
axH3.plot(x3, hRecursion[3], 'c', label = 'Recursion')
axH3.plot(x3, hExplicit[3], 'b:', label = 'Explicit')
axH3.set_ylim(-25,25)
axH3.set_xlim(-3,3)
axH3.grid()
axH3.set_title('H$_3$')
axH3.legend()

figH4, axH4 = plt.subplots()
figH4.set_size_inches(9,9)
axH4.plot(x3, hRecursion[4], 'c', label = 'Recursion')
axH4.plot(x3, hExplicit[4], 'b:', label = 'Explicit')
axH4.set_ylim(-25,25)
axH4.set_xlim(-3,3)
axH4.grid()
axH4.set_title('H$_4$')
axH4.legend()

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
figP.set_size_inches(9,9)
for n in range(nMaxP):
    text = 'P$_%i$' % n
    axP.plot(x4, pRecursion[n], label = text)
axP.set_ylim(-1,1.05)
axP.set_xlim(-1,1)
axP.grid()
axP.set_title('Legendre Polynomials')
axP.legend()

# 5 - Compute numerical derivatives

x5 = 10

f = x5**4

hlist = []
for i in range(10):
    n = 10**(-i)
    hlist.append(n)

h1 = np.linspace(100,1, endpoint = False)
h2 = np.linspace(1,0.01, endpoint = False)
h3 = np.linspace(0.01,0.0001, endpoint = False)
h4 = np.linspace(0.0001,0.000001, endpoint = False)
h5 = np.linspace(0.000001,0.00000001)

#h = np.asarray(hlist)
h = np.hstack((h1,h1,h2,h3,h4,h5))

for i in h:
    df2p = ((x5+h)**4 - (x5-h)**4)/(2*h)
    df4p = ((x5-2*h)**4 - 8*(x5-h)**4 + 8*(x5+h)**4 - (x5+2*h)**4)/(12*h)
    d2f2p = ((x5+h)**4 - 2*((x5)**4) + (x5-h)**4)/(h**2)

df2pUAE = np.abs(df2p - 4000)
df4pUAE = np.abs(df4p - 4000)
d2f2pUAE = np.abs(d2f2p - 1200)

figdf2p, axdf2p = plt.subplots()
figdf2p.set_size_inches(9,9)
axdf2p.plot(h, df2pUAE)
axdf2p.set_title('First Derivative: 2-Point Central Finite Difference', fontsize = 15)
axdf2p.set_xscale('log')
axdf2p.set_yscale('log')
axdf2p.grid()
axdf2p.set_xlabel('log$_{10}$h', fontsize = 15)
axdf2p.set_ylabel('$log_{10}\Delta df$', fontsize = 15)
#axdf2p.set_xlimit()

figdf4p, axdf4p = plt.subplots()
figdf4p.set_size_inches(9,9)
axdf4p.plot(h, df4pUAE)
axdf4p.set_xscale('log')
axdf4p.set_yscale('log')
axdf4p.set_title('First Derivative: 4-Point Central Finite Difference', fontsize = 15)
axdf4p.grid()
axdf4p.set_xlabel('log$_{10}$h', fontsize = 15)
axdf4p.set_ylabel('$log_{10}\Delta df$', fontsize = 15)

figd2f2p, axd2f2p = plt.subplots()
figd2f2p.set_size_inches(9,9)
axd2f2p.plot(h, d2f2pUAE)
axd2f2p.set_yscale('log')
axd2f2p.set_xscale('log')
axd2f2p.set_title('Second Derivative: 2-Point Central Finite Difference', fontsize = 15)
axd2f2p.grid()
axd2f2p.set_xlabel('log$_{10}$h', fontsize = 15)
axd2f2p.set_ylabel('$log_{10}\Delta d^2f$', fontsize = 15)