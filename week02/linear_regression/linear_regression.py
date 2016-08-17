# -*- coding: utf-8 -*-
import numpy as np

# Computing the cost J
def computeCost(X, y, theta):
    m = len(y)
    J = 1/(2 * m) * np.sum(np.power(np.dot(X, theta) - y, 2))
    return J

# Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for iter in range(num_iters):
        theta = theta - alpha * (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J = computeCost(X, y, theta)
        J_history.append(J)
    return theta, J_history
