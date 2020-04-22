import numpy as np

def compute_dropout(A, keep_prob):
    D = np.random.rand(*A.shape)
    D = (D < keep_prob).astype(int)
    A_after_droput = A * D
    #escala el valor esperado de A_prev que fue modificado por el dropout
    A_after_droput = A_after_droput / keep_prob
    return A_after_droput, D


def L2_regularization(lambd, parameters, L, m):
    l2 = 0
    #de 1 hasta L
    for l in range(L + 1):
        l2 = l2 + np.sum(np.square(parameters['W'+str(l)]))
    regularization = (lambd / 2*m) * l2

    return regularization