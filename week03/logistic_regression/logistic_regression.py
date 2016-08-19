# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def safe_log(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def compute_cost(theta, X, y):
    """
    Compute the cost J
    
    Args:
        theta (np.array n x 1) : parameters of hypothesis function
        X (np.array m x n)     : input dataset
        y (np.array m x 1)     : output dataset
    Returns:
        float : cost value
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(-y * safe_log(h) - (1-y) * safe_log(1-h))
    return J

def compute_grad(theta, X, y):
    """
    Compute the gradients of the cost
    
    Args:
        theta (np.array n x 1) : parameters of hypothesis function
        X (np.array m x n)     : input dataset
        y (np.array m x 1)     : output dataset
    Returns:
        np.array n x 1 : gradient values
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = 1/m * np.dot(X.T, (h - y))
    return grad

def compute_theta(initial_theta, X, y):
    def _compute_cost(theta, X, y):
        t = theta.reshape(len(theta), 1)
        J = compute_cost(t, X, y)
        return J
    
    def _compute_grad(theta, X, y):
        t = theta.reshape(len(theta), 1)
        grad = compute_grad(t, X, y)
        return grad.flatten()
    
    theta = optimize.fmin_cg(_compute_cost, fprime=_compute_grad,
                             x0=initial_theta, args=(X, y))
    return theta
