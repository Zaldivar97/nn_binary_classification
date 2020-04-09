import numpy as np
from enum import Enum
from .activations import sigmoid, relu


def __linear_forward(A, W, b):
    Z: np.ndarray = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation: Enum):
    Z, linear_cache = __linear_forward(A_prev, W, b)
    if activation == Activation.SIGMOID:
        A, activation_cache = sigmoid(Z)
    elif activation == Activation.RELU:
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    L = parameters // 2
    A_prev = X
    caches = []
    # de 1 hasta L-1
    for l in range(1, L):
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)],
                                             parameters['b'+str(l)], Activation.RELU)
        caches.append(cache)
        A_prev = A
    A, cache = linear_activation_forward(A_prev, parameters['W'+str(L)],
                                         parameters['b'+str(L)], Activation.SIGMOID)
    caches.append(cache)

    return A, caches
    


class Activation(Enum):
    SIGMOID = 1
    RELU = 2
