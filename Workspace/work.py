import numpy as np

def mse(y_true, y_pred):
    sub = np.multiply(y_true - y_pred, y_true - y_pred)
    return np.mean(sub)

def fit_ols(X,y):
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)
