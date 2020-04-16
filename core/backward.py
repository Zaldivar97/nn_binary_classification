import numpy as np
from enum import Enum
from .forward import Activation
from .activations import relu_backward, sigmoid_backward


def __linear_backward(dZ, D, linear_cache, keep_prob, lambd):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m  # derivada de W
    if lambd > 0:
        dW = dW + (lambd/m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m  # derivada de b
    dA_prev = np.dot(W.T, dZ) * D
    dA_prev = dA_prev / keep_prob

    return dA_prev, dW, db


def linear_backward_activation(dA, cache_tuple, activation: Enum, keep_prob, lambd):
    linear_cache, activation_cache, D = cache_tuple
    if activation == Activation.SIGMOID:
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == Activation.RELU:
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = __linear_backward(dZ, D, linear_cache, keep_prob, lambd)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, keep_prob = 1, lambd = 0):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    # derivada del costo respecto a AL
    dA = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    current_layer_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = linear_backward_activation(
                                                        dA, current_layer_cache, Activation.SIGMOID, keep_prob)
    # de 0 a L-2
    for l in reversed(range(L-1)):
        current_layer_cache = caches[l]
        dA_prev, dW, db = linear_backward_activation(grads['dA'+str(l+1)], current_layer_cache
                                                                            , Activation.RELU, keep_prob)
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db

    return grads                                                                         