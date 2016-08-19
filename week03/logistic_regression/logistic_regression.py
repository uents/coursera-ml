# -*- coding: utf-8 -*-
import numpy as np

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
    J = 1/m * sum(-y * safe_log(h) - (1-y) * safe_log(1-h))
    return J[0]

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



