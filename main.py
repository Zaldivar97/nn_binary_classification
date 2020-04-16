import numpy as np

def normalize_input(X):
    X_mean = np.mean(X) #promedio
    X = X - X_mean  #Xi - promedio
    X_variance = np.var(X) 
    X = X / X_variance # ( X1 - promedio ) / varianza

    return X