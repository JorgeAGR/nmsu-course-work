import numpy as np

def MLP(X,Wh,Wo):
    '''%-----------
    %INITIALIZE
    %-----------''';
    a = 1.7159
    b = 2/3.
    N = len(X[0,:])               #%number of training data pts
    bias = -1                  # %initial bias value
    O=[]
    X = np.concatenate((bias*np.ones([1,N]) , X),axis=0)    #%add zero'th order terms
    H=[]
    for j in range(len(Wh)):
        H.append(len(Wh[j]))
    
    
    '''
    %-----
    %MAIN
    %-----

    %-------------------------------------------------
    %PROPAGATE INPUTS FORWARD
    %-------------------------------------------------
    %------------------------
    %HIDDEN LAYER
    %------------------------''';
    
    for j in range(len(H)):               #%loop over each hidden layer
        if j==0:
            V = Wh[j]@X               #%weighted sum of inputs [1] Eqn(4.29/30)
        else:
            V = Wh[j]@O[j-1]          #%weighted sum of hidden inputs [1] Eqn(4.29/31)
        
        PHI = a * np.tanh(b*V)     #%acivation function [1] Eqn(4.37)
        O.append( np.concatenate((bias*np.ones([1,N]),PHI),axis=0))   #%add zero'th order terms
    

    '''%------------------------
    %OUTPUT LAYER
    %------------------------''';
    V = Wo@O[-1]            #%weighted sum of inputs [1] Eqn(4.29)
    Y = a * np.tanh(b*V)    #%activation function [1] Eqn(4.37)

    return Y
