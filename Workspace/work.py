import numpy as np
import random
import statistics
import math
 

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)
 

def fit_ols(X,y):
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)
 

def fit_gradient_descent(X, y, learning_rate, epochs):
    w = np.array([[0],[0],[0]])
    n = len(y)
    for i in range(0,epochs):
        y_hat = X @ w
        print("MSE: ", mse(y, y_hat))
        r = y_hat - y
        deriv_J = 2/n*X.T @ r
        w = w - learning_rate*deriv_J
    return w
 

def train_test_split(X, y, test_fraction):
    train_numbers = random.sample(range(len(y)), int(len(y)*(1-test_fraction)))
    test_numbers = list(set(range(len(y))) - set(train_numbers))
    train_X = X[train_numbers]
    test_X = X[test_numbers]
    train_y = y[train_numbers]
    test_y = y[test_numbers]
    return train_X, test_X, train_y, test_y
 

def fit_ridge(X, y, lam):
    A = X.T @ X + lam * np.eye(len(X[0]))
    b = X.T @ y
    return np.linalg.solve(A, b)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def binary_cross_entropy(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))


def fit_logistic_regression(X, y, lr, epochs, reg_lambda=0.0):
    w = np.zeros(X.shape[1])
    b = 0
    n = len(y)
    loss_history = []
    for i in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        L_total = binary_cross_entropy(y, y_hat) + reg_lambda* w.T @ w
        loss_history.append(L_total)
        grad = 1/n * X.T @ (y_hat - y) + 2*reg_lambda*w
        bias_grad = np.mean(y_hat - y)
        w = w - lr*grad
        b = b - lr*bias_grad
    return w, b, loss_history


def main():
    np.random.seed(0)
    n = 200
    X0 = 2*np.random.randn(n//2, 2) + np.array([-2, -2])
    X1 = 2*np.random.randn(n//2, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.concatenate([
        np.zeros(n//2),
        np.ones(n//2)
    ])

    w, b, loss_history = fit_logistic_regression(
        X,
        y,
        lr=0.1,
        epochs=1000
    )

    print(w)
    print(b)
    print(loss_history[0])
    print(loss_history[-1])

    # lam = [0.0001, 0.001, 0.1, 1, 10, 100]
    # rng = np.random.default_rng(seed=26)
    # x1 = rng.normal(0,3, size=(1000,1))
    # x2 = rng.normal(0,3, size=(1000,1))
    # x3 = x2 + rng.normal(0, 0.001, size=(1000,1))
    # test_X = np.hstack([x1, x2, x3])
    # w_true = np.array([[2], [5], [1]])
    # error = rng.normal(loc=0.0, scale=0.01, size=(1000,1))
    # test_y = test_X @ w_true + error
    # weights_by_lambda = [[] for _ in range(6)]
    # mse_by_lambda = [[] for _ in range(6)]
    # wols = []
    # mse_ols = []
    # for i in range(100):
    #     x1 = rng.normal(0,3, size=(20,1))
    #     x2 = rng.normal(0,3, size=(20,1))
    #     x3 = x2 + rng.normal(0, 0.001, size=(20,1))
    #     train_X = np.hstack([x1, x2, x3])
    #     w_true = np.array([[2], [5], [1]])
    #     error = rng.normal(loc=0.0, scale=0.01, size=(20,1))
    #     train_y = train_X @ w_true + error
    #     w_ols = fit_ols(train_X, train_y)
    #     wols.append(w_ols)
    #     mse_ols.append(mse(test_y, test_X @ w_ols))
    #     for j in lam:
    #         w_ridge = fit_ridge(train_X,train_y, j)
    #         if j == 0.0001:
    #             mse_by_lambda[0].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[0].append(w_ridge)
    #         if j == 0.001:
    #             mse_by_lambda[1].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[1].append(w_ridge)
    #         if j == 0.1:
    #             mse_by_lambda[2].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[2].append(w_ridge)
    #         if j == 1:
    #             mse_by_lambda[3].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[3].append(w_ridge)
    #         if j == 10:
    #             mse_by_lambda[4].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[4].append(w_ridge)
    #         if j == 100:
    #             mse_by_lambda[5].append(mse(test_y, test_X @ w_ridge))
    #             weights_by_lambda[5].append(w_ridge)
    # print("Mean MSE of lambda 0.0001: ", statistics.mean(mse_by_lambda[0]))
    # print("Mean MSE of lambda 0.001: ", statistics.mean(mse_by_lambda[1]))
    # print("Mean MSE of lambda 0.1: ", statistics.mean(mse_by_lambda[2]))
    # print("Mean MSE of lambda 1: ", statistics.mean(mse_by_lambda[3]))
    # print("Mean MSE of lambda 10: ", statistics.mean(mse_by_lambda[4]))
    # print("Mean MSE of lambda 100: ", statistics.mean(mse_by_lambda[5]))
    # print("Standard Deviation of lambda 0.0001: ", statistics.stdev(mse_by_lambda[0]))
    # print("Standard Deviation of lambda 0.001: ", statistics.stdev(mse_by_lambda[1]))
    # print("Standard Deviation of lambda 0.1: ", statistics.stdev(mse_by_lambda[2]))
    # print("Standard Deviation of lambda 1: ", statistics.stdev(mse_by_lambda[3]))
    # print("Standard Deviation of lambda 10: ", statistics.stdev(mse_by_lambda[4]))
    # print("Standard Deviation of lambda 100: ", statistics.stdev(mse_by_lambda[5]))
    # print("Mean MSE of OLS: ", statistics.mean(mse_ols))
    # print("Standard Deviation of OLS: ", statistics.stdev(mse_ols))
    # count = 0
    # for i in lam:
    #     stacked_w = np.hstack(weights_by_lambda[count])
    #     column_variance = np.var(stacked_w, axis=1, keepdims=True)
    #     print("Variance of Ridge w1,w2,w3: ", column_variance, "for lambda: ", i)
    #     count = count + 1
    # stacked_w = np.hstack(wols)
    # column_variance = np.var(stacked_w, axis=1, keepdims=True)
    # print("Variance of OLS w1,w2,w3: ", column_variance)
    # n = len(y)
    # eigenvalues = np.linalg.eigvalsh(X.T @ X + lam * np.eye(len(X[0])))
    # max = np.max(eigenvalues)
    # min = np.min(eigenvalues)
    # limit = n/max
    # K = max/min
    # print("w: ", w_ridge)
    # print("w norm: ", np.linalg.norm(w_ridge))
    # print("Limit and K: ", limit, K)
    # print("Training MSE: ", mse(train_y, train_X @ w_ridge))
    # print("Test MSE: ", mse(test_y, test_X @ w_ridge))
    # w_GD = fit_gradient_descent(train_X, train_y, 0.01, 200)
    # print("w_true, w_ols, w_GD: ", w_true, w_ols, w_GD)
    # print("Test MSE for OLS: ", mse(test_y, test_X @ w_ols))
    # print("Test MSE for GD: ", mse(test_y, test_X @ w_GD))
    # print("Train MSE for OLS: ", mse(train_y, train_X @ w_ols))
    # print("Train MSE for GD: ", mse(train_y, train_X @ w_GD))

if __name__ == "__main__":
    main()