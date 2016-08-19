# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../logistic_regression')
import logistic_regression as lr


def map_feature(X1, X2):
    """
    Mapping function to polynomial features
    
    Args:
        X1 (np.array m x n)     : input feature 1
        X2 (np.array m x n)     : input feature 2
    Returns:
        np.array m x n : new feature array X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    m = X1.shape[0];
    degree = 6
    out = np.ones((m, 1))
    for i in range(1, degree+1):
        for j in range (0, i + 1):
            column = (np.power(X1, i-j) * np.power(X2, j)).reshape((m, 1))
            out = np.c_[out, column]
    return out

def compute_cost_reg(theta, X, y, lam):
    """
    Compute the cost with reguralization
    
    Args:
        theta (np.array n x 1) : parameters of hypothesis function
        X (np.array m x n)     : input dataset
        y (np.array m x 1)     : output dataset
        lam (float)            : reguralization coefficient
    Returns:
        float : cost value
    """
    m = len(y)
    J = lr.compute_cost(theta, X, y) + lam/m * np.sum(np.power(theta[1:], 2))
    return J

def compute_grad_reg(theta, X, y, lam):
    """
    Compute the gradients of the cost
    
    Args:
        theta (np.array n x 1) : parameters of hypothesis function
        X (np.array m x n)     : input dataset
        y (np.array m x 1)     : output dataset
        lam (float)            : reguralization coefficient
    Returns:
        np.array n x 1 : gradient values
    """
    m = len(y)
    grad = lr.compute_grad(theta, X, y)
    grad[1:] = grad[1:] + lam/m * theta[1:]
    return grad

def compute_theta(initial_theta, X, y, lam):
    def _compute_cost(theta, X, y, lam):
        t = theta.reshape(len(theta), 1)
        J = compute_cost_reg(t, X, y, lam)
        return J
    
    def _compute_grad(theta, X, y, lam):
        t = theta.reshape(len(theta), 1)
        grad = compute_grad_reg(t, X, y, lam)
        return grad.flatten()
    
    theta = optimize.fmin_cg(_compute_cost, fprime=_compute_grad,
                             x0=initial_theta, args=(X, y, lam))
    return theta
