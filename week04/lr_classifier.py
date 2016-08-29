# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

import sys,os
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
    + '/../week03/regularized_logistic_regression')
import regularized_logistic_regression as lr


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def optimize_thetas(X, y, num_labels, lmd):
    """
    Optimize the parameters
    
    Parameters
    ----------
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output label dataset
    num_labels : int
        numeber of classification labels
    lmd : float
        regularization coefficient
    
    Returns
    -------
    thetas : array-like, shape (n_dim, 1)
        optimized parameters of each class
    """
    assert(X.shape[0] == y.shape[0])

    def _compute_cost(theta, X, y, lmd):
        _t = theta.reshape(len(theta), 1)
        _y = y.reshape(len(y), 1)
        J = lr.compute_cost_reg(_t, X, _y, lmd)
        return J

    def _compute_grad(theta, X, y, lmd):
        _t = theta.reshape(len(theta), 1)
        _y = y.reshape(len(y), 1)
        grad = lr.compute_grad_reg(_t, X, _y, lmd)
        return grad.ravel()

    m = X.shape[0]
    n = X.shape[1]

    thetas = np.zeros((num_labels, n + 1))
    X = np.c_[np.ones((m, 1)), X]
    initial_theta = np.zeros((n + 1, 1))

    for label in range(1, num_labels+1):
        _y = np.array([1 if c == label else 0 for c in y])
        print('training : label', label, '...')
        res = optimize.minimize(
            method='BFGS', fun=_compute_cost, jac=_compute_grad,
            x0=initial_theta, args=(X, _y, lmd), options={'maxiter': 50})
        thetas[label-1] = res.x

    return thetas


def predict(thetas, X):
    """
    predict classification
    
    Parameters
    ----------
    thetas : array-like, shape (n_classes, n_dim+1)
        parameters
    X : array-like, shape (n_examples, n_dim)
        input dataset
    
    Returns
    -------
    ypreds : array-like, shape (n_examples, 1)
        prediction classes and values
    """
    X = np.c_[np.ones((X.shape[0], 1)), X]
    assert(X.shape[1] == thetas.shape[1])

    values = sigmoid((np.dot(X, thetas.T)))
    preds = [{'class': (np.argmax(p) + 1), 'value': np.max(p)} for p in values]
    return preds

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
