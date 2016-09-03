# -*- coding: utf-8 -*-
import numpy as np

def cost_function(theta, X, y):
    """
    Cost function
    
    Parameters
    ----------
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset

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

def gradient_descent(theta, X, y, alpha, num_iters):
    """
    Compute the gradients of cost function
    
    Parameters
    ----------
    theta : array-like, shape (n_dim, 1)
        initial parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    alpha : float
        learning coefficient
    num_iters : int
        number of iterations
    
    Returns
    -------
    theta : array-like, shape (n_dim, 1)
        optimized parameters of hypothesis function
    J_history : list (n_iterations)
        history of cost value
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))

    J_history = []
    for iter in range(num_iters):
        theta = theta - alpha * (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J = cost_function(theta, X, y)
        J_history.append(J)
    return theta, J_history

def predict(theta, X):
    """
    Predict regression

    Parameters
    ----------
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset

    Returns
    -------
    ypreds : array-like, shape (n_examples, 1)
        predicted dataset
    """
    return np.dot(X, theta)
