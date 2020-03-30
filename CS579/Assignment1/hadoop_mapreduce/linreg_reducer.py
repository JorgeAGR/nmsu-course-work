import sys
import numpy as np

for i, line in enumerate(sys.stdin):
    key, value = line.split('\t',1)
    xTx, xTy = value.split(']]',1)
    xTx += ']]'
    xTy = xTy.rstrip('\n').strip('[').strip(']')
    
    xTy = np.fromstring(xTy, sep=',')
    num_params = len(xTy)
    xTy = np.reshape(xTy, (num_params, 1))
    
    xTx = xTx.replace('[','').replace(']','')
    xTx = np.fromstring(xTx, sep=',')
    xTx = np.reshape(xTx, (num_params, num_params))
    
    if i == 0:
        xTx_sum = np.zeros(xTx.shape)
        xTy_sum = np.zeros(xTy.shape)

    xTx_sum += xTx
    xTy_sum += xTy
 
xTx_sum_inv = np.linalg.inv(xTx_sum)
W = xTx_sum_inv @ xTy_sum

print(W)