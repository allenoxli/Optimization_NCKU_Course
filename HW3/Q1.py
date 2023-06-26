# In[]:

import matplotlib.pyplot as plt

import numpy as np


x = np.array([0.4, 1, 1.5, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
y = np.array([28, 31, 30, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32])


def find_linear_variable(x, y):
    r"""Linear equation y = mx + b.
    To find m and b, we can minimize f(m, b) = (mx + b - y)**2.
    """
    n = len(x)
    m = (x.sum()*y.sum() - n*(x*y).sum() )/ (x.sum()**2 - n*(x**2).sum())
    b = (y.sum() - m*x.sum()) / n
    return m, b

def find_polynomial_variable(x, y, degree=2):
    r"""Quadratic regression equation y = a0 + a1*x + a2*x**2.
    To find a0, a1 and a2, we can minimize f(a0, a1, a2) = (a0 + a1*x + a2*x**2 - y)**2.
    """
    matrix = [x**i for i in range(0, degree+1)]
    X = np.array(matrix).T

    coef = np.linalg.inv(X.T @ X) @ X.T @ y[:, None]

    # a0, a1, a2, ... , ak (k = number of degree)
    return coef

def find_equ_val(x, coef):
    coef = coef.squeeze()
    matrix = [x**i for i in range(0, len(coef))]
    X = np.array(matrix).T
    return (X * coef).sum(axis=-1)

quar_coef = find_polynomial_variable(x, y, degree=2)
linear_coef = find_linear_variable(x, y)
linear_coef2 = find_polynomial_variable(x, y, degree=1)
print(quar_coef, '\n')
print(linear_coef, '\n')
print(linear_coef2, '\n')

y_pred = find_equ_val(x, quar_coef)
fit_line = np.linspace(x.min(), x.max(), 100)
plt.plot(fit_line, find_equ_val(fit_line, linear_coef2))

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
