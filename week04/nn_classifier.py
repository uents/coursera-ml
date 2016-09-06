# -*- coding: utf-8 -*-
import numpy as np

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../week03/')
from common import *


def predict(theta1, theta2, X):
    """
    predict classification
    
    Parameters
    ----------
    theta1 : array-like, shape (n_hidden_unit, n_dim+1)
        parameters of input layer
    theta2 : array-like, shape (n_classes, n_hidden_unit+1)
        parameters of hidden layer
    X : array-like, shape (n_examples, n_dim)
        input dataset
    
    Returns
    -------
    ypreds : list (n_examples)
        prediction classes and values
    """
    assert(X.shape[1] + 1 == theta1.shape[1])
    assert(theta1.shape[0] + 1 == theta2.shape[1])

    m = X.shape[0]
    a1 = np.c_[np.ones((m,1)), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones((m,1)), sigmoid(z2)]
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    ps = [{'class': np.argmax(p) + 1, 'value': np.max(p)} for p in a3]
    preds = [{'class': 0, 'value': p['value']} if p['class'] == 10 else p for p in ps]
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
