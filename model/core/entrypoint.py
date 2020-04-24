import numpy as np
from .initializers import initialize, Initializer
from .forward import Activation, L_model_forward
from .backward import L_model_backward
from .cost import cost, update_parameters


def normalize_input(X):
    X_mean = np.mean(X)  # promedio
    X = X - X_mean  # Xi - promedio
    X_variance = np.var(X)
    X = X / X_variance  # ( X1 - promedio ) / varianza

    return X


def nn_model(X_train, Y_train, layer_dims, learning_rate=0.005, epocs=2000, keep_prob=1, lambd=0, initializer_type=Initializer.HE_INITIALIZER, debug = True):
    parameters = initialize(layer_dims, initializer_type)
    X_train_normalized = normalize_input(X_train)
    cost_cache = [] #para graficar el costo
    for i in range(epocs):
        AL, caches_list, dropout_cache = L_model_forward(X_train_normalized, parameters, keep_prob)
        if len(dropout_cache) == 0:
            dropout_cache = None
        cost_value = cost(AL, Y_train, lambd, parameters)
        cost_cache.append(cost_value)

        grads = L_model_backward(AL, Y_train, caches_list, dropout_cache, keep_prob, lambd)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if debug:
            if i % 50 == 0:
                print(f'Costo en la iteracion {i}: {cost_value}')    
    return parameters