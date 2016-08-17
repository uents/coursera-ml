# -*- coding: utf-8 -*-
import numpy as np

# Computing the cost J
def compute_cost(X, y, theta):
    m = len(y)
    J = 1/(2 * m) * np.sum(np.power(np.dot(X, theta) - y, 2))
    return J

# Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for iter in range(num_iters):
        theta = theta - alpha * (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J = compute_cost(X, y, theta)
        J_history.append(J)
    return theta, J_history
