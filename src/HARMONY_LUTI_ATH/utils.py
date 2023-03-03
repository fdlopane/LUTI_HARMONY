"""
utils.py
Data building utilities
"""
import numpy as np
import pickle
import struct

"""
Load a numpy matrix from a file
"""
def loadMatrix(filename):
    with open(filename,'rb') as f:
        matrix = pickle.load(f)
    return matrix

###############################################################################

"""
Save a numpy matrix to a file
"""
def saveMatrix(matrix,filename):
    with open(filename,'wb') as f:
        pickle.dump(matrix,f)

###############################################################################

"""
Load a QUANT format matrix into python.
A QUANT matrix stores the row count (m), column count (n) and then m x n IEEE754 floats (4 byte) of data
"""
def loadQUANTMatrix(filename):
    with open(filename,'rb') as f:
        (m,) = struct.unpack('i', f.read(4))
        (n,) = struct.unpack('i', f.read(4))
        print("loadQUANTMatrix::m=",m,"n=",n)
        matrix = np.arange(m*n,dtype=np.float).reshape(m, n) #and hopefully m===n, but I'm not relying on it
        for i in range(0,m):
            data = struct.unpack('{0}f'.format(n), f.read(4*n)) #read a row back from the stream - data=list of floats
            for j in range(0,n):
                matrix[i,j] = data[j]
    return matrix

###############################################################################






