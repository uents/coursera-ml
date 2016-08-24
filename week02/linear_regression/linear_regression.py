# -*- coding: utf-8 -*-
import numpy as np

def compute_cost(X, y, theta):
    """
    Compute the cost value
    
    Parameters
    ----------
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function

    Returns
    -------
    J : float
        cost value
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))
    
    J = 1/(2*m) * np.sum(np.power(np.dot(X, theta) - y, 2))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Compute the gradients of cost function
    
    Parameters
    ----------
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function
    alpha : float
        learning coefficient
    num_iters : int
        number of iterations
    
    Returns
    -------
    grad : array-like, shape (n_dim, 1)
        gradient values
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))

    J_history = []
    for iter in range(num_iters):
        theta = theta - alpha * (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J = compute_cost(X, y, theta)
        J_history.append(J)
    return theta, J_history
