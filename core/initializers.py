import numpy as np
HE_INITIALIZER = "h"
XAVIER_INITIALIZER = "x"


def initialize_W(layer_dims, initializer_type):
    if initializer_type == HE_INITIALIZER:
        initializer = __he_initializer
    elif initializer_type == XAVIER_INITIALIZER:
        initializer = __xavier_initializer
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],
                                                 layer_dims[l-1]) * initializer(layer_dims[l])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def __normal_init():
    return 0.01


def __xavier_initializer(number_incoming_neurons):
    return np.sqrt(1/number_incoming_neurons)


def __he_initializer(number_incoming_neurons):
    return np.sqrt(2/number_incoming_neurons)
