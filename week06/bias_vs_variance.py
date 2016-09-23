# -*- coding: utf-8 -*-
import numpy as np

def cost_function(theta, X, y, lmd):
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
    lmd : float
        regularization coefficient

    Returns
    -------
    J : float
        cost value
    D : array-like, shape (n_dim, 1)
        gradient of cost function
    """
    m = y.shape[0]
    n = theta.shape[0]
    assert(X.shape == (m, n))
    assert(y.shape[1] == 1)
    assert(theta.shape[1] == 1)

    h = predict(theta, X)
    J = 1/(2*m) * np.sum(np.power(h - y, 2))
    J += lmd/(2*m) * np.sum(np.power(theta[1:], 2))
    D = 1/m * np.dot(X.T, h - y)
    D[1:] += lmd/m * theta[1:]
    return J, D

def optimize_params(initial_theta, X, y, lmd):
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
    lmd : float
        regularization coefficient
    
    Returns
    -------
    theta : array-like, shape (n_dim, 1)
        optimized parameters of hypothesis function
    J_history : list (n_iterations)
        history of cost value
    """
    J_history = []; J = 0.

    def _cost_function(theta, X, y, lmd):
        nonlocal J
        t = theta.reshape(-1, 1)
        J, D = cost_function(t, X, y, lmd)
        return J, D.ravel()

    def _callback(theta):
        nonlocal J_history, J
        J_history.append(J)

    from scipy.optimize import minimize
    res = minimize(method='CG', fun=_cost_function, jac=True,
                   x0=initial_theta, args=(X, y, lmd),
                   options={'maxiter': 200}, callback=_callback)

    return res.x, J_history


def predict(theta, X):
    """
    Predict regression

    Parameters
    ----------
    theta : array-like, shape (n_dim, 1)
        parameters of hypothesis function
    X : array-like, shape (n_examples, n_dim)
        input dataset

    Returns
    -------
    ypreds : array-like, shape (n_examples, 1)
        predicted dataset
    """
    return np.dot(X, theta).reshape(-1, 1)


def train_linear_regression(X, y, lmd):
        initial_theta = np.zeros(X.shape[1])
        theta, J_history = optimize_params(initial_theta, X, y, lmd)
        return theta.reshape(-1, 1), J_history

def learning_curve(Xtrain, ytrain, Xcv, ycv, lmd):
    m = ytrain.shape[0]
    Jtrain = [None for i in range(m+1)]
    Jcv = [None for i in range(m+1)]

    for i in range(1,m+1):
        _Xtrain = Xtrain[0:i,:]
        _ytrain = ytrain[0:i]
        theta, J_history = train_linear_regression(_Xtrain, _ytrain, lmd)
        Jtrain[i], D = cost_function(theta, _Xtrain, _ytrain, 0.)
        Jcv[i], D = cost_function(theta, Xcv, ycv, 0.)

    return Jtrain, Jcv

def poly_features(X, p):
    Xpoly = np.zeros((X.shape[0], p))
    for i in range(p):
        Xpoly[:,i] = np.power(X, i+1)
    return Xpoly

def feature_standardize(X, mu=None, sigma=None):
    m, p = X.shape

    if mu is None:
        mu = [None for x in range(p)]
        for i in range(p):
            mu[i] = np.mean(X[:,i])
        mu = np.array(mu).reshape(1, -1)

    if sigma is None:
        sigma = [None for x in range(p)]
        for i in range(p):
            sigma[i] = np.std(X[:,i])
        sigma = np.array(sigma).reshape(1, -1)

    Xstd = np.divide(np.subtract(X, mu), sigma)
    return Xstd, mu, sigma

def validation_curve(Xtrain, ytrain, Xcv, ycv, lmds):
    results = []

    for lmd in lmds:
        theta, J_history = train_linear_regression(Xtrain, ytrain, lmd)
        Jtrain, D = cost_function(theta, Xtrain, ytrain, 0.)
        Jcv, D = cost_function(theta, Xcv, ycv, 0.)
        results.append({'lambda' : lmd, 'train_error' : Jtrain, 'cv_error' : Jcv})

    return results
