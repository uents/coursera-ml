# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np

def savefig(file_name):
    os.makedirs('assets', exist_ok=True)
    file_path = os.path.join('assets', file_name)
    plt.savefig(file_path)
    print("saved", file_path)

def safe_log(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

