import matplotlib.pyplot as plt

import numpy as np
from math import sin


x = np.array([0.4, 1, 1.5, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
y = np.array([28, 31, 30, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32])


def MSE(y, y_pred):
    return ((y_pred - y)**2).sum() / len(y)

def RMSE(y, y_pred):
    return MSE(y, y_pred)**(1/2)

def find_polynomial_variable(x, y, degree=2):
    r"""Quadratic regression equation y = a0 + a1*x + a2*x**2.
    To find a0, a1 and a2, we can minimize f(a0, a1, a2) = (a0 + a1*x + a2*x**2 - y)**2.
    """
    matrix = [x**i for i in range(0, degree+1)]
    matrix.append([sin(val) for val in x])
    X = np.array(matrix).T

    coef = np.linalg.inv(X.T @ X) @ X.T @ y[:, None]

    # a0, a1, a2, ... , ak (k = number of degree)
    return coef

def find_equ_val(x, coef):
    coef = coef.squeeze()
    matrix = [x**i for i in range(0, len(coef)-1)]
    matrix.append([sin(val) for val in x])
    X = np.array(matrix).T
    return (X * coef).sum(axis=-1)

coef = find_polynomial_variable(x, y, degree=1)
y_pred = find_equ_val(x, coef)
error = RMSE(y, y_pred)
print(coef)
print(error)
