import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd

'''
References:
http://123.physics.ucdavis.edu/week_0_files/taylor_181-199.pdf
J. Mathews and R.L. Walker, Mathematical Methods of Physics, 2nd Ed. pp 387-393
'''

def lsf(x, y, sx, sy, n=1):
	
    X = np.zeros(shape = (n+1, 1)) # Data Vector (Outputs)
    M = np.zeros(shape = (n+1, n+1)) # Measurement Matrix (Inputs)
    a = np.zeros(shape = (n+1, 1)) # Coeffficients Vector
    counter = 0
    while True:
        counter += 1
        dydx = np.zeros(len(x))
        A = a.copy()
        for i in range(len(a)):
            dydx += a[i] * i * x ** (i-1)#(i) * x ** (i-1)
            w = 1 / (sy ** 2 + (dydx * sx)**2)
        for i in range(n+1):
            for j in range(n+1):
                M[i][j] = np.sum(w * x**(j+i))
            X[i][0] = np.sum(w * y * x ** i)
        a = np.dot(np.linalg.inv(M), X)
        if (np.abs(A - a).all() < 1e-12) or (counter == 100):
            break
    
    a = a.flatten()
    da = np.sqrt(np.linalg.inv(M).diagonal())
    	
    def func(a, x):
        f = 0
        for i in range(len(a)):
            f += a[i] * x ** (i)
        return f
    
    chisq = np.sum((y - func(a, x))**2 * w)
    rechisq = chisq / (len(x) - 2)
    
    print(a, da, rechisq)
    return a, da, rechisq
	
class LSF(object):
    
    def __init__(self, x, y, sx, sy, fit='linear', n=1):
        
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
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
            self.w = 1 / (self.sy ** 2 + (self.dfunc(x) * self.sx)**2)
            for i in range(self.n+1):
                for j in range(self.n+1):
                    M[i][j] = np.sum(self.w * x**(j+i))
                X[i][0] = np.sum(self.w * y * x ** i)
            self.a = np.dot(np.linalg.inv(M), X).flatten()
            if (np.abs(A - self.a).all() < 1e-12) or (counter == 100):
                break
        
        self.a = self.a.flatten()
        self.da = np.sqrt(np.linalg.inv(M).diagonal())
    '''    
    def fit_trig(self): # Only finds them if frequency known!
        X = np.zeros(shape = (self.n+1, 1))
        M = np.zeros(shape = (self.n+1, self.n+1))
        self.a = np.zeros(self.n+1)
        counter = 0
        while True:
            counter += 1
            A = self.a.copy()
            self.w = 1 / (self.sy ** 2 + (self.dfunc(x) * self.sx)**2)
            for i in range(self.n+1):
                for j in range(self.n+1):
                    if i != j:
                        M[i][j] = np.sum(self.w * np.cos(f*x) * np.sin(f*x))
                        X[i][0] = np.sum(self.w * y * np.sin(f*x))
                    elif (i == j == 1):
                        M[i][j] = np.sum(self.w * np.cos(f*x)**2)
                        X[i][0] = np.sum(self.w * y * np.cos(f*x))
                    else:
                        M[i][j] = np.sum(self.w * np.sin(x)**2)
            self.a = np.dot(np.linalg.inv(M), X).flatten()
            if (np.abs(A - self.a).all() < 1e-12) or (counter == 100):
                break
            self.a = self.a.flatten()
            self.da = np.sqrt(np.linalg.inv(M).diagonal())
    '''
        
    def evaluate(self):
        chisq = np.sum((y - self.func(x))**2 * self.w)
        self.rechisq = chisq / (len(x) - 2)
        
    def func(self, x):
        def linear():
            return self.a[0] + self.a[1] * x
        
        def poly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * x ** (i)
            return f
        
        def trig():
            return self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        funcs = {'linear': linear,
                'poly': poly,
                'trig': trig,}
        return funcs.get(self.fit)()
    
    def dfunc(self, x):
        def dlinear():
            return self.a[1]
        
        def dpoly():
            f = 0
            for i in range(len(self.a)):
                f += self.a[i] * i * x ** (i-1)
            return f
        
        def dtrig():
            return -self.a[0] * np.sin(x) + self.a[1] * np.cos(x)
        
        dfuncs = {'linear': dlinear,
                  'poly': dpoly,
                  'trig': dtrig,}
        return dfuncs.get(self.fit)()
'''
    def cmu_fit(self):
        # For documentation purposes. Translation of the macro made
        # by CMU
    	try:
    		if not sx:
    			sx = np.zeros(len(x))
    	except:
    		pass
    	A = 0
    	while True:
    		a = A
    		w = 1/(sy**2 + (A*sx)**2)
    		S = np.sum(w)
    		Sx = np.sum(w * x)
    		Sy = np.sum(w * y)
    		Sxx = np.sum(w * x**2)
    		Sxy = np.sum(w * x * y)
    		delta = S * Sxx - Sx ** 2
    		A = (S * Sxy - Sx * Sy) / delta
    		if np.abs(A-a) < 1e-10:
    			break
    	B = (Sxx * Sy - Sx * Sxy) / delta
    	sA = np.sqrt(S / delta)
    	sB = np.sqrt(Sxx / delta)
    	chisq = np.sum((y - (A*x + B))**2 * w)
    	rechisq = chisq / (len(x) - 2)
'''

def f(x):
	return 5*np.cos(x)
'''
def residual(vars, x, data, eps_data):
	
	a0 = vars[0]
	a1 = vars[1]
	#a2 = vars[2]
	
	model = a0 + a1 * x #+ a2 * x ** 2 
	
	return (data - model) / eps_data
'''

x = np.random.uniform(0, 4, 10)
sx = np.random.uniform(-0.01, 0.01, 10)
#sx = np.zeros(10)
y_real = f(x)
sy = np.random.uniform(-1, 1, 10)
#sy = np.ones(10)
y = y_real + sy

'''
df = pd.read_csv('CMU_LSF3.csv')
x = df['X data'].values
sx = df['X error'].values
y = df['Y data'].values
sy = df['Y error'].values
'''
lsffit = LSF(x, y, sx, sy, fit='poly', n=1)
#coeff, dcoeff, rechisq = lsf(x, y, sx, sy, n=1)
#npcoeff = np.polyfit(x, y, 3)
#spcoeff, _ = leastsq(residual, [0, 1], args = (x, y, sy))

#fit = lambda x: coeff[0] + coeff[1] * x #+ coeff[2] * x ** 2 + coeff[3] * x ** 3
#npfit = np.poly1d(npcoeff)
#spfit = lambda x: spcoeff[0] + spcoeff[1] * x + spcoeff[2] * x ** 2

linspace = np.linspace(0,4)

#plt.plot(linspace, f(linspace), 'o')
plt.errorbar(x, y, xerr = sx, yerr = sy, fmt = '.')
#plt.plot(linspace, fit(linspace), '--', label = 'My')
plt.plot(linspace, lsffit.func(linspace), label = 'LSF')
#plt.plot(linspace, npfit(linspace), '.-', label = 'Np')
#plt.plot(linspace, spfit(linspace), '--', label = 'Sp')
plt.legend()
plt.show()
