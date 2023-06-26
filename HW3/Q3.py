import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def MSE(y, y_pred):
    return ((y_pred - y)**2).sum() / len(y)

def RMSE(y, y_pred):
    return MSE(y, y_pred)**(1/2)

x = np.array([0.4, 1, 1.5, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
y = np.array([28, 31, 30, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32])

# x_fit = np.linspace(x.min(), x.max(), 300).reshape(-1,1)


# intercepts, coef = [], []

# plt.figure(figsize=(15, 10))
# error = []
# minimum = 100000
# degree_range = (2, 16)
# tar_degree = 0
# for degree in range(*degree_range):
#     poly = PolynomialFeatures(degree=degree, include_bias=False)
#     poly.fit_transform(x[:, None], y)

#     poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     poly_model.fit(x[:, np.newaxis], y)

#     y_fit = poly_model.predict(x_fit)
#     y_pred = poly_model.predict(x[:, None])

#     err_tmp = MSE(y, y_pred)
#     error.append(err_tmp)

#     if minimum > err_tmp:
#         minimum = err_tmp
#         tar_degree = degree

#     linear_model = poly_model.named_steps['linearregression']
#     print(linear_model.coef_ )

#     coef.append(linear_model.coef_)
#     intercepts.append(linear_model.intercept_)

#     plt.subplot(121)
#     plt.plot(x_fit, y_fit, label=f"degree {degree}" )



# plt.legend(loc='best')
# plt.scatter(x, y)

# print(f'The minimum error is {minimum} at degree {tar_degree}')
# plt.subplot(122)

# assert len(range(*degree_range)) == len(error)
# plt.plot(range(*degree_range), error)
# plt.show()

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

quar_coef = find_polynomial_variable(x, y, degree=1)
y_pred = find_equ_val(x, quar_coef)
error = RMSE(y, y_pred)
print(error)

degree_range = (1, 11)
minimum = 100000
error_list = []
tar_degree = 0
for degree in range(*degree_range):
    # poly_model = np.poly1d(np.polyfit(x, y, degree))
    # y_pred = poly_model(x)

    coef = find_polynomial_variable(x, y, degree)
    y_pred = find_equ_val(x, coef)

    error = RMSE(y, y_pred)

    error_list.append(error)
    if minimum > error:
        minimum = error
        tar_degree = degree

    # plt.scatter(x, y)
    # fit_line = np.linspace(x.min(), x.max(), 100)
    # plt.plot(fit_line, poly_model(fit_line))

# plt.show()
print(error_list)

plt.plot(range(*degree_range), error_list)
plt.xlabel ('Order')
plt.ylabel ('Error')
plt.show()

# print(minimum)
print(f'The minimum error is {minimum} at degree {tar_degree}')

print('|order|error|')
print('|-|-|')
for i in range(len(error_list)):
    print(f'|{i}|{error_list[i]}|')

print()
print('|order', end='|')
for i in range(len(error_list)):
    print(f'{i+1}', end='|')
print()
print(f'|-'*(len(error_list)+1), end='|\n')
print('|error', end='|')
for i in range(len(error_list)):
    print(f'{error_list[i]:.3f}', end='|')

