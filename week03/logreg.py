# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common')
from common import *


def cost_function(theta, X, y):
    """
    Cost function and gradient
    
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
    D : array-like, shape (n_dim, 1)
        gradient
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))
    
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(-y * safe_log(h) - (1-y) * safe_log(1-h))
    D = 1/m * np.dot(X.T, (h - y))
    return J, D

def optimize_params(initial_theta, X, y):
    """
    Optimize the parameters of hypothesis function
    
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
    J_history : list (n_iterations)
        history of cost value
    """
    J_history = []; J = 0.
    
    def _cost_function(theta, X, y, J_history):
        nonlocal J
        t = theta.reshape(-1, 1)
        J, D = cost_function(t, X, y)
        return J, D.ravel()

    def _callback(theta):
        nonlocal J_history, J
        J_history.append(J)

    J_history = []
    res = scipy.optimize.minimize(
        method='BFGS', fun=_cost_function, jac=True,
        x0=initial_theta, args=(X, y, J_history),
        options={'maxiter': 400}, callback=_callback)

    return res.x, J_history
