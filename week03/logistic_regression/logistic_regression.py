# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def safe_log(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def compute_cost(theta, X, y):
    """
    Compute the cost value
    
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
    
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(-y * safe_log(h) - (1-y) * safe_log(1-h))
    return J

def compute_grad(theta, X, y):
    """
    Compute the gradients of cost function
    
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
    grad : array-like, shape (n_dim, 1)
        gradient values
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))

    h = sigmoid(np.dot(X, theta))
    grad = 1/m * np.dot(X.T, (h - y))
    return grad

def optimize_theta(initial_theta, X, y):
    """
    Optimize the parameters of hypothesis function with gradient descent
    
    Parameters
    ----------
    initial_theta : array-like, shape (n_dim, 1)
        initial parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    
    Returns
    -------
    theta : array-like, shape (n_dim, 1)
        optimized parameters of hypothesis function
    J : float
        cost value
    """
    def _compute_cost(theta, X, y):
        t = theta.reshape(len(theta), 1)
        J = compute_cost(t, X, y)
        return J
    
    def _compute_grad(theta, X, y):
        t = theta.reshape(len(theta), 1)
        grad = compute_grad(t, X, y)
        return grad.ravel()
    
    res = optimize.minimize(
        method='BFGS', fun=_compute_cost, jac=_compute_grad,
        x0=initial_theta, args=(X, y), options={'maxiter': 400})

    return res.x, res.fun
