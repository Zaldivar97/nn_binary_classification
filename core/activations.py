import numpy as np
from initializers import 

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[ Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = sigmoid(Z)
    dZ = dA * s * (s - 1)

    return dZ