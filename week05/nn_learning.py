# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../week03/')
from common import *
from logreg_regular import *


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def rand_initialize_weights(shape, epsilon_init=0.12):
    return np.random.rand(shape[0], shape[1]) * 2 * epsilon_init - epsilon_init

def cost_function(theta1, theta2, X, y, lmd):
    """
    Cost function and gradient
    
    Parameters
    ----------
    theta1 : array-like, shape (n_hidden_unit, n_dim+1)
        parameters of input layer
    theta2 : array-like, shape (n_classes, n_hidden_unit+1)
        parameters of hidden layer
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    
    Returns
    -------
    J : float
        cost value
    D1 : array-like, shape (n_hidden_unit, n_dim+1)
        gradient of parameter of input layer
    D2 : array-like, shape (n_classes, n_hidden_unit+1)
        gradient of parameter of hidden layer
    """
    assert(X.shape[0] == y.shape[0])
    assert(X.shape[1] + 1 == theta1.shape[1])
    assert(theta1.shape[0] + 1 == theta2.shape[1])

    m, n = X.shape
    num_labels = len(np.unique(y))

    # outputs with 1-of-K notation
    Y = (np.identity(num_labels)[y,:])[:,0,:]

    # forward propagation
    m = X.shape[0]
    a1 = np.c_[np.ones((m,1)), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones((m,1)), sigmoid(z2)]
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    assert(Y.shape == a3.shape)

    # cost value
    J = (1/m * np.sum(-Y * safe_log(a3) - (1 - Y) * safe_log(1 - a3))) + \
        (lmd/(2*m) * (np.sum(np.power(theta1[:,1:], 2)) + \
                      np.sum(np.power(theta2[:,1:], 2))))

    # backpropagation
    delta3 = a3 - Y
    delta2 = np.dot(delta3, theta2[:,1:]) * sigmoid_gradient(z2)
    Delta3 = np.dot(delta3.T, a2)
    Delta2 = np.dot(delta2.T, a1)    

    # gradients
    D1 = 1/m * Delta2
    D1[:,1:] += lmd/m * theta1[:,1:]
    D2 = 1/m * Delta3
    D2[:,1:] += lmd/m * theta2[:,1:]

    return J, D1, D2


def optimize_params(initial_theta1, initial_theta2, X, y, lmd):
    """
    Optimize the parameters
    
    Parameters
    ----------
    initial_theta1 : array-like, shape (n_hidden_units, n_dim+1)
        initial parameters of input layer
    initial_theta2 : array-like, shape (n_labels, n_hidden_units+1)
        initial parameters of hidden layer
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output label dataset
    lmd : float
        regularization coefficient
    
    Returns
    -------
    theta1 : array-like, shape (n_hidden_units, n_dim+1)
        optimized parameters of input layer
    theta2 : array-like, shape (n_labels, n_hidden_units+1)
        optimized parameters of hidden layer
    J_history : list (n_iterations)
        history of cost value
    """

    def _reshape_theta(thetas, num_input_unit, num_hidden_unit, num_labels):
        brk = num_hidden_unit * (num_input_unit + 1)
        theta1 = thetas[0:brk].reshape(num_hidden_unit, num_input_unit + 1)
        theta2 = thetas[brk:].reshape(num_labels, num_hidden_unit + 1)
        return theta1, theta2

    def _cost_function(thetas, num_input_unit, num_hidden_unit, X, y, lmd):
        nonlocal J
        num_labels = len(np.unique(y))
        t1, t2 = _reshape_theta(thetas, num_input_unit,
                                num_hidden_unit, num_labels)
        J, D1, D2 = cost_function(t1, t2, X, y, lmd)
        return J, np.r_[D1.ravel(), D2.ravel()]

    def _callback(thetas):
        nonlocal J_history, J
        J_history.append(J)

    J_history = []; J = 0
    
    num_labels = len(np.unique(y))
    num_input_unit = initial_theta1.shape[1] - 1
    num_hidden_unit = initial_theta2.shape[1] - 1
    initial_thetas = np.r_[initial_theta1.ravel(), initial_theta2.ravel()]

    args = (num_input_unit, num_hidden_unit, X, y, lmd)
    res = optimize.minimize(
        method='CG', fun=_cost_function, jac=True,
        x0=initial_thetas, args=args,
        options={'maxiter': 50, 'disp': True}, callback=_callback)

    theta1, theta2 = _reshape_theta(
        res.x, num_input_unit, num_hidden_unit, num_labels)
    return theta1, theta2, J_history


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
    ypreds : array-like, shape (n_examples, 1)
        prediction classes and values
    """
    assert(X.shape[1] + 1 == theta1.shape[1])
    assert(theta1.shape[0] + 1 == theta2.shape[1])

    m = X.shape[0]
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    preds = [{'class': np.argmax(p), 'values': p} for p in a3]
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
