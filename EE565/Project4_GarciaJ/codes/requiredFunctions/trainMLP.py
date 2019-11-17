import numpy as np
import numpy.matlib
from requiredFunctions.MLP import MLP

def trainMLP(X,D,H,eta,alpha,epochMax,MSETarget=1e-12, verbose=True, X_val=None, D_val=None):
    '''%==========================================================================
    % Call Syntax:  [Wh,Wo,MSE] = trainMLP(X,D,H,eta,alpha,epochMax,MSETarget)
    %
    % Description:  The matrix implementation of the Backpropagation algorithm
    %               for a Multilayer Perceptron (MLP).
    %
    % Input Arguments:
    %	Name: X
    %	Type: (p x N) dimensional matrix, where p is a number of the inputs and N is a training size
    %	Description: a series of input oberservation vectors as columns in a matrix
    %
    %	Name: D
    %	Type: (m x N) dimensional matrix, where m is a number of the output neurons and N is a training size
    %	Description: a series of desired output responses as columns in a matrix
    %
    %	Name: H
    %	Type: vector
    %	Description: Number of hidden neurons in each hidden layer
    %
    %	Name: eta
    %	Type: scalar
    %	Description: learning rate parameter
    %
    %	Name: alpha
    %	Type: scalar
    %	Description: momentum parameter
    %
    %	Name: epochMax
    %	Type:  scalar
    %	Description: maximum number of training epochs
    %
    %	Name: MSETarget (optional)
    %	Type:  scalar (default: MSETarget = 1e-12)
    %	Description: target mse error rate
    %
    % Output Arguments:
    %	Name: Wh
    %	Type: Cell arrray, with each cell containing a (H(j) x p+1) dimensional matrix, where H(j) is the number of neurons in the j'th hidden layer
    %	Description: hidden layer weight matrix for each hidden layer
    %
    %	Name: Wo
    %	Type: (m x H(end)+1) dimensional matrix
    %	Description: Output layer weight matrix
    %
    %	Name: MSE
    %	Type: vector
    %	Description: mean square error per epoch
    %
    %--------------------------------------------------------------------------
    % Notes:
    %
    % References:
    % [1] - S. Haykin, Neural Networks and Learning Machines, vol. 3, Pearson,2009
    %
    %--------------------------------------------------------------------------
    % Author: Steven Sandoval
    %--------------------------------------------------------------------------
    % Revision History:
    %
    %          Steven Sandoval - 25 September 2012 - Adapted 2-layer MLP codes by Marcelo Augusto Costa Fernandes (mfernandes@dca.ufrn.br).
    %          Steven Sandoval -   06 October 2012 - Cleaned code, added comments
    %          Steven Sandoval -      01 July 2018 - updated references and variable names, added ReLU activation function
    %          Steven Sandoval -   30 October 2019 - simplified code for EE565 Project
    %          Brandon Byford  -   30 OCtober 2019 - ported to python3 
    %==========================================================================
    ''';
    
    '''%-----------
    %INITIALIZE
    %-----------''';
    
    a = 1.7159
    b = 2/3.
    [p, N] = np.shape(np.array(X))                 #dimension of input vector and number of training data pts
    m = len(D)                                  #number of output neurons
    bias = -1                                      #bias value
    Wh=[]
    WhAnt=[]
    X = np.concatenate([bias*np.ones([1,N]),X ],axis=0)                  #add zero'th order terms
    for j in range(len(H)):
        if j ==0:
            Wh.append(np.random.rand(H[j],p+1))                          #initialize first hidden layer weights
            WhAnt.append(np.zeros([H[j],p+1]))                      #initialize variable for weight correction using momentum 
        else:
            Wh.append( np.random.rand(H[j],H[j-1]+1)  ) #initialize hidden layer weights
            WhAnt.append(np.zeros([H[j],H[j-1]+1]) )                #initialize variable for weight correction using momentum 
            
    Wo = np.random.rand(m,H[-1]+1)                                 #initialize output layer weights
    WoAnt = np.zeros([m,H[-1]+1])                            #initialize variable for weight correction using momentum
    MSETemp = np.zeros([epochMax,1])                   #allocate memory for MSE error for each epoch
    MSETemp_val = np.zeros([epochMax,1])
    for i in range(epochMax):
        O=[]
        '''%-------------------------------------------------
        %PROPAGATE INPUTS FORWARD
        %-------------------------------------------------''';
        
        '''%------------------------
        %HIDDEN LAYER
        %------------------------''';
        for j in range(len(H)):               #%loop over each hidden layer
            if j==0:
                V = Wh[j]@X               #%weighted sum of inputs [1] Eqn(4.29/30)
            else:
                V = Wh[j]@O[j-1]          #%weighted sum of hidden inputs [1] Eqn(4.29/31)
            PHI = a * np.tanh(b*V)         #%activation function [1] Eqn(4.37)
            O.append(np.concatenate([bias*np.ones([1,N]),PHI],axis=0))   #%add zero'th order terms
        
        '''%------------------------
        %OUTPUT LAYER
        %------------------------''';
        V = Wo@O[-1]                 #%weighted sum of inputs [1] Eqn(4.29)
        Y = a * np.tanh(b*V)       #%activation function [1] Eqn(4.37)


        '''%------------------------
        %ERROR CALCULATION 
        %------------------------''';
        E = D - Y                  #%calclate error
        mse = np.mean(E**2)    #%calculate mean square error
        
        MSETemp[i,0] = mse           #%save mse
        
        '''%------------------------
        %Validation - Jorge Garcia
        %------------------------'''
        if np.all(X_val) and np.all(D_val):
            Y_val = MLP(X_val, Wh, Wo)
            E_val = D_val - Y_val
            mse_val = np.mean(E_val**2)
            MSETemp_val[i,0] = mse_val


        #%DISPLAY PROGRESS, BREAK IF ERROR CONSTRAINT MET
        if verbose:
            print('epoch = ' +str(i)+ ' mse = ' +str(mse))
        if (mse < MSETarget):
            MSE = MSETemp
            if np.all(X_val) and np.all(D_val):
                MSE_val = MSETemp_val
                return(Wh,Wo,MSE,MSE_val)
            else:
                return(Wh,Wo,MSE)
        
        '''%-------------------------------------------------
        %BACK PROPAGATE ERROR
        %-------------------------------------------------

        %------------------------
        %OUTPUT LAYER
        %------------------------''';
        PHI_PRMo = b/a *(a-Y)*(a+Y)   #%derivative of activation function [1] Eqn(4.38)

        dGo = E * PHI_PRMo                 #%local gradient [1] Eqn(4.35/39)
        DWo = dGo@O[-1].T                    #%non-scaled weight correction [1] Eqn(4.27)

        Wo = Wo + eta*DWo + alpha*WoAnt  #%weight correction including momentum term [1] Eqn(4.41)
        WoAnt = eta*DWo + alpha*WoAnt                         #%save weight correction for momentum calculation

        '''%------------------------
        %HIDDEN LAYERS
        %------------------------''';
        
        
        for j in np.arange(len(H))[::-1]:
            PHI_PRMh = b/a *(a-O[j])*(a+O[j])         #%derivative of activation function [1] Eqn(4.38)
            
            if j==(len(H)-1):
                dGh = PHI_PRMh * (Wo.T @ dGo)                   #%local gradient[1] Eqn(4.36/40)
            else:
                dGh = PHI_PRMh * (Wh[j+1].T @ np.matlib.repmat( dGo,Wh[j+1].shape[0],1 ) )         # %local gradient[1] Eqn(4.36/40)
            dGh = dGh[1:,:]                             #%dicard first row of local gradient (bias doesn't update)
            
            if j==0:
                DWh = dGh@X.T                            #%non-scaled weight correction [1] Eqn(4.27/30)
            else:
                DWh = dGh@O[j-1].T                       #%non-scaled weight correction [1] Eqn(4.27/31)
            
            Wh[j] =Wh[j]+ eta*DWh + alpha*WhAnt[j]  # %weight correction including momentum term [1] Eqn(4.41)
            WhAnt[j] =eta*DWh + alpha*WhAnt[j]     #%save weight correction for momentum calculation

    MSE = MSETemp
    if np.all(X_val) and np.all(D_val):
        MSE_val = MSETemp_val
        return(Wh,Wo,MSE,MSE_val)
    else:
        return(Wh,Wo,MSE)
