import numpy as np
from enum import Enum
from .forward import Activation
from .activations import relu_backward, sigmoid_backward


def __linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m  # derivada de W
    db = np.sum(dZ, axis=1, keepdims=True) / m  # derivada de b
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_backward_activation(dA, cache_tuple, activation: Enum):
    linear_cache, activation_cache = cache_tuple
    if activation == Activation.SIGMOID:
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == Activation.RELU:
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = __linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    # derivada del costo respecto a AL
    dA = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    current_layer_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = linear_backward_activation(
                                                        dA, current_layer_cache, Activation.SIGMOID)
    for l in reversed(range(L-1)):
        current_layer_cache = caches[l]
        dA_prev, dW, db = linear_backward_activation(grads['dA'+str(l+1)], current_layer_cache
                                                                            , Activation.RELU)
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db

    return grads                                                                         


