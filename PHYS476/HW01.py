import numpy as np
import matplotlib.pyplot as plt
import sympy as sym



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

fig3b = plt.figure()

plt.subplot2grid((2,1), (0,0))
for n in range(nMaxH):
    plt.plot(x3, hRecursion[n])
plt.ylim(-25, 25)
plt.xlim(-3,3)
plt.grid()
plt.title('Recursion H$_n$')

plt.subplot2grid((2,1), (1,0))
for n in range(nMaxH):
    plt.plot(x3, hExplicit[n], '--')
plt.ylim(-25, 25)
plt.xlim(-3,3)
plt.grid()
plt.title('Explicit H$_n$')

fig3c = plt.figure()

plt.subplot2grid((2,1),(0,0))
plt.plot(x3, hRecursion[3], 'c')
plt.plot(x3, hExplicit[3], 'b:')
plt.ylim(-25,25)
plt.xlim(-3,3)
plt.grid()
plt.title('H$_3$')

plt.subplot2grid((2,1),(1,0))
plt.plot(x3, hRecursion[4], 'c')
plt.plot(x3, hExplicit[4], 'b:')
plt.ylim(-25,25)
plt.xlim(-3,3)
plt.grid()
plt.title('H$_4$')

plt.show()

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

fig4, ax4 = plt.subplots()
for n in range(nMaxP):
    ax4.plot(x4, pRecursion[n])
ax4.set_ylim(-1,1.05)
ax4.set_xlim(-1,1)
ax4.grid()
ax4.set_title('Legendre Polynomials')

# 5 - Compute numerical derivatives

x5 = 10

f = x5**4

h = np.linspace(1, 0, num = 250, endpoint = False) 
for i in h:
    df2p = ((x5+h)**4 - (x5-h)**4)/(2*h)
    df4p = ((x5-2*h)**4 - 8*(x5-h)**4 + 8*(x5+h)**4 - (x5+2*h)**4)/(12*h)