import numpy as np
from enum import Enum
from .activations import sigmoid, relu
from .regularization import compute_dropout


def __linear_forward(A, W, b):
    Z = np.dot(W, A) + b
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


def L_model_forward(X, parameters, keep_prob):

    L = len(parameters) // 2
    A = X
    caches = []
    dropout_cache = []
    # de 1 hasta L-1
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)],
                                             parameters["b"+str(l)], Activation.RELU)
        if keep_prob < 1:
            A, D = compute_dropout(A, keep_prob)
            dropout_cache.append(D)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)],
                                         parameters["b"+str(L)], Activation.SIGMOID)
    caches.append(cache)

    return AL, caches, dropout_cache
    

class Activation(Enum):
    SIGMOID = 1
    RELU = 2
