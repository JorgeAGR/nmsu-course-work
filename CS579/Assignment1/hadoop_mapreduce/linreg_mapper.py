import sys
import numpy as np
#id,date,price,bedrooms,bathrooms
#sqft_living,sqft_lot,floors,waterfront
#view,condition,grade,sqft_above,sqft_basement
#yr_built,yr_renovated,zipcode,lat,long
#sqft_living15,sqft_lot15

# Number of entries in data
n_rows = 21614

def array2string(array):
    elements = False
    if array.shape[1] == 1:
        elements = True
    array = list(array)
    for i, element in enumerate(array):
        if elements:
            array[i] = float(element)
        else:
            array[i] = list(element)
    
    return array

for i, line in enumerate(sys.stdin):
    if i == 0:
        continue
    line = line.strip()
    key = line.split(',')[0].strip('"')
    line = line.split(',')[2:]

    y = np.zeros((n_rows,1))
    y[i] = line[0]

    x = np.zeros((n_rows,len(line[1:])))
    x[i,:] = list(map(lambda x: float(x.strip('"')), line[1:]))
    xT = x.T
    xTx = xT @ x
    xTy = xT @y
    
    print('{}\t{}{}'.format(key, array2string(xTx), array2string(xTy)))
