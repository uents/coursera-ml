# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/')
from common import *
from logreg import *


def map_feature(X1, X2):
    """
    Mapping function to polynomial features
    
    Parameters
    ----------
    X1 : array-like, shape (n_examples, 1)
        input feature 1
    X2 : array-like, shape (n_examples, 1)
        input feature 2

    Returns:
    ----------
    X : array-like, shape (n_exampls, n_new_dim)
        nolynomial features X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    assert(X1.shape == X2.shape)
    
    m = X1.shape[0];
    degree = 6
    out = np.ones((m, 1))
    for i in range(1, degree+1):
        for j in range (0, i + 1):
            column = np.power(X1, i-j) * np.power(X2, j)
            out = np.c_[out, column]
    return out

def cost_function_reg(theta, X, y, lmd):
    """
    Cost function and gradient with regularization
    
    Parameters
    ----------
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    lmd : float
        regularization coefficient

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
    
    J, D = cost_function(theta, X, y)
    J += lmd/m * np.sum(np.power(theta[1:], 2))
    D[1:] += lmd/m * theta[1:]
    return J, D

def optimize_params_reg(initial_theta, X, y, lmd):
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
    lmd : float
        regularization coefficient
    
    Returns
    -------
    theta : array-like, shape (n_dim, 1)
        optimized parameters of hypothesis function
    J_history : list (n_iterations)
        history of cost value
    """
    def _cost_function(theta, X, y, lmd, J_history):
        t = theta.reshape(-1, 1)
        J, D = cost_function_reg(t, X, y, lmd)
        J_history.append(J)
        return J, D.ravel()

    J_history = []
    res = scipy.optimize.minimize(
        method='BFGS', fun=_cost_function, jac=True,
        x0=initial_theta, args=(X, y, lmd, J_history),
        options={'maxiter': 400})

    return res.x, J_history

def predict(theta, X):
    """
    Predict classification

    Parameters
    ----------
    thetas : array-like, shape (n_dim, 1)
        parameters
    X : array-like, shape (n_examples, n_dim)
        input dataset
    
    Returns
    -------
    ypreds : array-like, shape (n_examples, 1)
        predicted classes
    """
    assert(X.shape[1] == theta.shape[0])

    preds = [1 if s >= 0.5 else 0 for s in sigmoid(np.dot(X, theta))]
    return np.array(preds).reshape(-1, 1)

def compute_train_accuracy(ypreds, y):
    """
    Compute the training accuracy
    
    Parameters
    ----------
    ypreds : array-like, shape (n_examples, 1)
        predicted dataset
    y : array-like, shape (n_examples, 1)
        correct dataset
    
    Returns
    -------
    accuracy : float
        training accuracy
    """
    assert(ypreds.shape == y.shape)
    return np.mean(ypreds.ravel() == y.ravel()) * 100.
