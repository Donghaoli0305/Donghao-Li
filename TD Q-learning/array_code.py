# ********
# This file is individualized to NetID dli106.
# ********
# No other imports are allowed
import numpy as np

def arith(x):
    # https://numpy.org/doc/stable/user/quickstart.html#basic-operations
    return 2 * np.power(x, 4) + 2 * x   

def agg(x):
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#calculation
    max_values = 3 * np.max(3 * x, axis=2) 
    min_value = 2 * np.min(max_values, axis=1) 
    return np.sum(min_value)   
    
def bool(x):
    return np.sum(np.power(x, 4) < (4 * x + 4))  

def bcast(x):
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    x1, x2 = x
    return (x1 + 4) * (3 * x2 - 3)   

def bcast_ax(x):
    # https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.newaxis
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.reshape.html
    x1, x2 = x
    return (4 + x1[:, np.newaxis, :]) * (3 * x2 - 3)   

def newax(x):
    x = np.array(x)
    return (x[:, np.newaxis, np.newaxis]**3) * (x[np.newaxis, :, np.newaxis]*2) * (3**x[np.newaxis, np.newaxis, :])   

def series_pow(x):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
    return np.sum(np.arange(x) ** 1.61)   

def series_alt(x):
    return np.sum((-1) ** np.arange(x) * (np.arange(x) ** 1.61))   

def series_dbl(x):
    x1, x2 = x
    i = np.arange(x1)
    j = np.arange(x2)
    return np.sum((3 * j + 4) * (i[:, np.newaxis] ** 4))   

def idx(x):
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    return 3 * x[::2] + 4 * np.square(x[1::2])   

def hypercube(x):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
    j = np.arange(2 ** x)
    i = np.arange(x)[:, np.newaxis]
    return (-1) ** ((j // (2 ** i)) % 2)   

