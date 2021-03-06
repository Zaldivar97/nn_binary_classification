import numpy as np


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
    dZ[ Z <= 0 ] = 0

    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)

    return dZ