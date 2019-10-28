# ## Gradient Descent Example
# ### Using : Python 3 , sympy , numpy , matplotlib

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#https://scipy-lectures.org/packages/sympy.html
# ^^ how to use sympy ^^ 

HX ,HY  = 50,50 #number of x,y  points for countour
xmin,xmax = -15,15
ymin,ymax = -12,12
x1 = np.linspace(xmin,xmax,HX)
x2 = np.linspace(ymin,ymax,HY)
X1,X2 = np.meshgrid(x1,x2) # genertate mesh grid
w1=sym.Symbol('w1') # define symbols
w2=sym.Symbol('w2')


bowl = 'bowl'
case2= ''
case = bowl #case select
if(case==bowl):
    j=(w1**2+w1*w2+3*w2**2)# define equation
elif(case==case2):
    #define other surfaces here
    pass
else:
    print('case not recognized')
    

#compute gradient
j_grad1=sym.diff(j,w1)
j_grad2=sym.diff(j,w2)
#compute hessian
hess11=sym.diff(j_grad1,w1)
hess12=sym.diff(j_grad1,w2)
hess21=sym.diff(j_grad2,w1)
hess22=sym.diff(j_grad2,w2)

#generate contour map
ConMap=np.zeros((HX,HY))
for i in range(HX):
    for k in range(HY):
        ConMap[i,k]=j.subs({w1:x1[i],w2:x2[k]})


w=[5,-5] # starting point
wStar=[0,0] # ending point
ew=[]
jw=[]
eta = 0.03 # learning rate
lim=50 # number of iterations
count= 0 
line=[]

while(True):
    #compute gradient matrix and hessian matrix
    g= np.array([float(j_grad1.subs({w1:w[0],w2:w[1]})),float(j_grad2.subs({w1:w[0],w2:w[1]}))])
    H= np.array([[float(hess11.subs({w1:w[0],w2:w[1]})),float(hess12.subs({w1:w[0],w2:w[1]}))],
                [float(hess21.subs({w1:w[0],w2:w[1]})),float(hess22.subs({w1:w[0],w2:w[1]}))]])
    
    wnew =  w-eta*g 
    #loop check
    if( count>lim ):
        print('Count Break')
        break
    elif(np.isnan(g).any()):
        print('nan break')
        break
    else:
        count=count +1
        wprev=w.copy()
        
        line.append(w)
        w=wnew.copy()
        
        ew.append(np.linalg.norm(w-wStar))
        jw.append(j.subs({w1:w[0],w2:w[1]}))
line=np.array(line)   

plt.figure(figsize=(10,10))
plt.contour(X1,X2,ConMap)
plt.plot(line[:,1],line[:,0])
plt.scatter(line[:,1],line[:,0])
plt.plot(wStar[1],wStar[0],'go')

plt.figure(figsize=(10,10))
plt.title('J(w) vs iterations')
plt.ylabel('J(w)')
plt.xlabel('iteration')
plt.plot(range(len(jw)),jw)

plt.figure(figsize=(10,10))
plt.title('E(w) vs iterations')
plt.ylabel('E(w)')
plt.xlabel('iteration')
plt.plot(range(len(ew)),ew)