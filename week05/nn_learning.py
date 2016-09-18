# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
from collections import namedtuple

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../common/')
from common import *


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def append_weights(theta1, theta2):
    return np.r_[theta1.ravel(), theta2.ravel()]

def divide_weights(thetas, spec):
    brk = spec.num_hid_units * (spec.num_in_units + 1)
    theta1 = thetas[0:brk].reshape(spec.num_hid_units, spec.num_in_units + 1)
    theta2 = thetas[brk:].reshape(spec.num_out_units, spec.num_hid_units + 1)
    return theta1, theta2

def rand_weights(shape, epsilon_init=0.12):
    return np.random.rand(shape[0], shape[1]) * 2 * epsilon_init - epsilon_init

def initialize_weights(spec):
    theta1 = rand_weights((spec.num_hid_units, spec.num_in_units+1))
    theta2 = rand_weights((spec.num_out_units, spec.num_hid_units+1))
    return append_weights(theta1, theta2)


def cost_function(thetas, spec, X, y, lmd):
    """
    Cost function and gradient
    
    Parameters
    ----------
    thetas : array-like, num_units
        weights of input and hidden layers
    spec : namedtuple
        layer specification
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output dataset
    
    Returns
    -------
    J : float
        cost value
    D : array-like, num_units
        gradient of weights of input and hidden layers
    """
    assert(X.shape[0] == y.shape[0])
    assert(X.shape[1] == spec.num_in_units)

    m, n = X.shape
    k = spec.num_out_units

    # device weights
    theta1, theta2 = divide_weights(thetas, spec)

    # outputs with 1-of-K notation
    Y = (np.identity(k)[y,:])[:,0,:]

    # forward propagation
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
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

    return J, append_weights(D1, D2)


def optimize_weights(initial_thetas, spec, X, y, lmd):
    """
    Optimize the weights
    
    Parameters
    ----------
    initial_thetas : array-like, num_units
        initial weights of input and hidden layers
    spec : namedtuple
        layer specification
    X : array-like, shape (n_examples, n_dim)
        input dataset
    y : array-like, shape (n_examples, 1)
        output label dataset
    lmd : float
        regularization coefficient
    
    Returns
    -------
    thetas : array-like, num_units
        optimized weights of input and hidden layers
    J_history : list, n_iterations
        history of cost value
    """

    def _cost_function(thetas, spec, X, y, lmd):
        nonlocal J
        J, D, = cost_function(thetas, spec, X, y, lmd)
        return J, D

    def _callback(thetas):
        nonlocal J_history, J
        J_history.append(J)

    J_history = []; J = 0
    
    res = scipy.optimize.minimize(
        method='CG', fun=_cost_function, jac=True,
        x0=initial_thetas, args=(spec, X, y, lmd),
        options={'maxiter': 50, 'disp': True}, callback=_callback)

    return res.x, J_history


def predict(thetas, spec, X):
    """
    predict classification
    
    Parameters
    ----------
    thetas : array-like, num_of_units
        weights of input and hidden layers
    X : array-like, shape (n_examples, n_dim)
        input dataset
    
    Returns
    -------
    ypreds : array-like, shape (n_examples, 1)
        prediction classes and values
    """
    assert(X.shape[1] == spec.num_in_units)

    m, n = X.shape
    theta1, theta2 = divide_weights(thetas, spec)

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


def compute_numerical_gradient(func, thetas, e=1e-4):
    numgrad = np.zeros(thetas.shape)
    perturb = np.zeros(thetas.shape)

    for p in range(len(thetas)):
        perturb[p] = e
        loss1, d1 = func(thetas - perturb)
        loss2, d2 = func(thetas + perturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad
