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
    
    def __init__(self, x, y, sx, sy, fit='poly', n=1):
        
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
        self.chisq = np.sum((self.y - self.func(self.x))**2 * self.w) # chisq matters too!
        self.rechisq = self.chisq / (len(self.x) - self.n + 1) # Check this? Apparent DOF is not just on x & y
        
        #WSSR - Weighted sum square residuals
        
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

def f(x):
	return 7 * x + 10

x = np.random.uniform(0, 4, 10)
#sx = np.random.uniform(-0.01, 0.01, 10)
sx = np.zeros(10)
y_real = f(x)
#sy = np.random.uniform(-1, 1, 10)
sy = np.ones(10)
y = y_real #+ sy

'''
df = pd.read_csv('CMU_LSF3.csv')
x = df['X data'].values
sx = df['X error'].values
y = df['Y data'].values
sy = df['Y error'].values
'''
lsffit = LSF(x, y, sx, sy, fit='poly', n=0)

linspace = np.linspace(0,4)

plt.legend()
plt.show()
