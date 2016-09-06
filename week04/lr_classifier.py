# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../week03/')
from common import *
from logreg_regular import *


def optimize_params(X, y, num_labels, lmd):
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

    def _cost_function(theta, X, y, lmd, j_hist):
        t = theta.reshape(-1, 1)
        J, D = cost_function_reg(t, X, y, lmd)
        j_hist.append(J)
        return J, D.ravel()

    m, n = X.shape
    num_labels = len(np.unique(y))

    X = np.c_[np.ones((m, 1)), X]
    thetas = np.zeros((num_labels, n + 1))
    initial_theta = np.zeros((n + 1, 1))
    J_history = {}

    for label in np.unique(y):
        _y = np.array([1 if l == label else 0 for l in y]).reshape(-1, 1)
        j_hist = []
        res = scipy.optimize.minimize(
            method='BFGS', fun=_cost_function, jac=True,
            x0=initial_theta, args=(X, _y, lmd, j_hist),
            options={'maxiter': 50})

        thetas[label] = res.x
        print('... training : label=' + str(label) + ' J=' + str(j_hist[-1]))
        J_history.update({str(label): j_hist})
    
    return thetas, J_history


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
    preds = [{'class': np.argmax(p), 'value': np.max(p)} for p in values]
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
