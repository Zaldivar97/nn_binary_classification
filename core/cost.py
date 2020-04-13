import numpy as np

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters

def cost(AL, Y):
    m = Y.shape[1]
    log_probs = np.multiply(Y,np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))
    cost = - np.sum(log_probs) / m

    return cost
