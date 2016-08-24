# -*- coding: utf-8 -*-
import numpy as np

import sys,os
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + '/../linear_regression')
import linear_regression as lr


def compute_cost_multi(X, y, theta):
    return lr.compute_cost(X, y, theta)

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    return lr.gradient_descent(X, y, theta, alpha, num_iters)
