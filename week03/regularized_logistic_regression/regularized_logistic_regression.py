# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

import sys,os
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + '/../logistic_regression')
import logistic_regression as lr


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def safe_log(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def map_feature(X1, X2):
    """
    Mapping function to polynomial features
    
    Parameters
    ----------
    X1 : array-like, shape (n_examples, n_dim)
        input feature 1
    X2 : array-like, shape (n_examples, n_dim)
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
            column = (np.power(X1, i-j) * np.power(X2, j)).reshape((m, 1))
            out = np.c_[out, column]
    return out

def compute_cost_reg(theta, X, y, lmd):
    """
    Compute the cost value with regularization
    
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
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))
    
    J = lr.compute_cost(theta, X, y) + lmd/m * np.sum(np.power(theta[1:], 2))
    return J

def compute_grad_reg(theta, X, y, lmd):
    """
    Compute the gradients of cost function with regularization
    
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
    grad : array-like, shape (n_dim, 1)
        gradient values
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))
    
    grad = lr.compute_grad(theta, X, y)
    grad[1:] = grad[1:] + lmd/m * theta[1:]
    return grad

def optimize_theta(initial_theta, X, y, lmd):
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
    J : float
        cost value
    """
    def _compute_cost(theta, X, y, lmd):
        t = theta.reshape(len(theta), 1)
        J = compute_cost_reg(t, X, y, lmd)
        return J
    
    def _compute_grad(theta, X, y, lmd):
        t = theta.reshape(len(theta), 1)
        grad = compute_grad_reg(t, X, y, lmd)
        return grad.ravel()
    
    res = optimize.minimize(
        method='BFGS', fun=_compute_cost, jac=_compute_grad,
        x0=initial_theta, args=(X, y, lmd), options={'maxiter': 400})

    return res.x, res.fun

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
    return np.array(preds).reshape(len(preds), 1)

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
