import numpy as np
from enum import Enum


class Initializer(Enum):
    HE_INITIALIZER = "h"
    XAVIER_INITIALIZER = "x"


def initialize(layer_dims, initializer_type: Enum):
    if initializer_type == Initializer.HE_INITIALIZER:
        initializer = __he_initializer
    elif initializer_type == Initializer.XAVIER_INITIALIZER:
        initializer = __xavier_initializer
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],
                                                 layer_dims[l-1]) * initializer(layer_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def __normal_init():
    return 0.01


def __xavier_initializer(number_incoming_neurons):
    return np.sqrt(1/number_incoming_neurons)


def __he_initializer(number_incoming_neurons):
    return np.sqrt(2/number_incoming_neurons)
