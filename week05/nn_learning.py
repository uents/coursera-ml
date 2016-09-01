# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

def safe_log(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def rand_initialize_weights(shape, epsilon_init=0.12):
    return np.random.rand(shape[0], shape[1]) * 2 * epsilon_init - epsilon_init

def compute_cost(theta1, theta2, num_labels, X, y, lmd):
    """
    compute the cost value
    
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
    """
    assert(X.shape[0] == y.shape[0])
    assert(X.shape[1] + 1 == theta1.shape[1])
    assert(theta1.shape[0] + 1 == theta2.shape[1])

    # outputs for 1-of-K notation
    Y = (np.identity(num_labels)[y-1,:])[:,0,:]

    # feedforward
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
    print("J =", J)

    # backpropagation
    delta3 = a3 - Y
    delta2 = np.dot(delta3, theta2[:,1:]) * sigmoid_gradient(z2)
    Delta3 = np.dot(delta3.T, a2)
    Delta2 = np.dot(delta2.T, a1)    

    # gradient for cost function
    theta1_grad = 1/m * Delta2 + \
                  lmd/m * np.c_[np.zeros((theta1.shape[0], 1)), theta1[:,1:]]
    theta2_grad = 1/m * Delta3 + \
                  lmd/m * np.c_[np.zeros((theta2.shape[0], 1)), theta2[:,1:]]

    """
    for k in range(m):
        xk = X[k,:].reshape(1,-1)
        yk = Y[k,:].reshape(1,-1)
        
        # feedforward and cost value
        a1 = np.c_[np.ones((1,1)), xk]
        z2 = np.dot(a1, theta1.T)
        a2 = np.c_[np.ones((1,1)), sigmoid(z2)]
        z3 = np.dot(a2, theta2.T)
        a3 = sigmoid(z3)
        assert(yk.shape == a3.shape)
        J += 1/m * np.sum(-yk * safe_log(a3) - (1 - yk) * safe_log(1 - a3))

        # backpropagation
        delta3 = a3 - yk
        delta2 = np.dot(delta3, theta2[:,1:]) * sigmoid_gradient(z2)
        Delta3 += np.dot(delta3.T, a2)
        Delta2 += np.dot(delta2.T, a1)
    """
    
    return J, theta1_grad, theta2_grad            


def optimize_thetas(initial_theta1, initial_theta2, \
                    num_labels, X, y, lmd):

    def _reshape_theta(thetas, num_input_unit, num_hidden_unit, num_labels):
        brk = num_hidden_unit * (num_input_unit + 1)
        theta1 = thetas[0:brk].reshape(num_hidden_unit, num_input_unit + 1)
        theta2 = thetas[brk:].reshape(num_labels, num_hidden_unit + 1)
        return theta1, theta2

    def _compute_cost(thetas, num_input_unit, num_hidden_unit, \
                      num_labels, X, y, lmd):
        _t1, _t2 = _reshape_theta(thetas, num_input_unit, \
                                  num_hidden_unit, num_labels)
        J, t1_grad, t2_grad = compute_cost(_t1, _t2, num_labels, X, y, lmd)
        return J, np.r_[t1_grad.ravel(), t2_grad.ravel()]

    num_input_unit = initial_theta1.shape[1] - 1
    num_hidden_unit = initial_theta2.shape[1] - 1
    initial_thetas = np.r_[initial_theta1.ravel(), initial_theta2.ravel()]
    _args = (num_input_unit, num_hidden_unit, num_labels, X, y, lmd)
             
    res = optimize.minimize(
        method='CG', fun=_compute_cost, jac=True,
        x0=initial_thetas, args=_args, options={'maxiter': 50, 'disp': True})

    return _reshape_theta(res.x, num_input_unit, num_hidden_unit, num_labels)

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
    a2 = sigmoid(z2)
    z3 = np.dot(np.c_[np.ones((m, 1)), a2], theta2.T)
    a3 = sigmoid(z3)

    preds = [{'class': (np.argmax(p) + 1), 'value': np.max(p)} for p in a3]
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
