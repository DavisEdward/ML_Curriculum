import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def fit_ols(X,y):
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)

def fit_gradient_descent(X, y, learning_rate, epochs):
    w = np.random.normal(loc=0.0, scale=2, size = (3,1))
    n = len(y)
    for i in range(0,epochs):
        y_hat = X @ w
        print("MSE: ", mse(y, y_hat))
        r = y_hat - y
        deriv_J = 2/n*X.T @ r
        w = w - learning_rate*deriv_J
    return w


def main():
    X = np.array([[3, 5, 4], [2, 2, 4], [7, 6, 9], [1, 2, 6], [4, 4, 4], [7, 9, 8], [3, 3, 6]])
    w_true = np.array([[2], [5], [1]])
    error = np.random.normal(loc=0.0, scale=0.5, size = (7,1))
    y = X @ w_true + error
    n = len(y)
    # w_ols = fit_ols(X,y)
    # w_GD = fit_gradient_descent(X, y, 0.01, 5000)
    # print("w_true, w_ols, w_GD: ", w_true, w_ols, w_GD)
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    max = np.max(eigenvalues)
    min = np.min(eigenvalues)
    limit = n/max
    K = max/min
    print("Limit and K: ", limit, K)


    




if __name__ == "__main__":
    main()