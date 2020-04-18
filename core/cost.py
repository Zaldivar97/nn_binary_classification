import numpy as np
from .regularization import L2_regularization

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters

def cost(AL, Y, lambd = 0, parameters = 0):
    m = Y.shape[1]
    l2_regularization = L2_regularization(lambd, parameters) if lambd > 0 else 0
    log_probs = np.multiply(Y,np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)) 
    cost = - np.sum(log_probs) / m
    cost = cost + l2_regularization

    return cost

