import numpy as np
from math import cos, sin, sqrt
import math
import random
import copy
import matplotlib.pyplot as plt


golden_ratio = (1 + sqrt(5)) / 2

def func(x):
    return x**2 * sin(x) * cos(x)

def f_minus(x):
    return -1*func(x)

def golden_search(func, lower, upper, tol=1e-5):
    r"""Golden-section search algorithm.
    Find the minimum of function `func` within the interval [`a`, `b`].
    """
    a, b = copy.deepcopy(lower), copy.deepcopy(upper)
    max_point, min_point = None, None
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    # Iterations.
    iter = 0
    min_list, max_list = [], []
    while abs(b - a) > tol:
        if func(c) > func(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

        max_point = (b + a) / 2
        max_list.append(func(max_point))


    a, b = copy.deepcopy(lower), copy.deepcopy(upper)
    while abs(b - a) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

        min_point = (b + a) / 2
        min_list.append(func(min_point))

    return min_point, max_point, min_list, max_list

def dichotomous_search(f, lower, upper, epsilon):
    a, b = copy.deepcopy(lower), copy.deepcopy(upper)
    x_list = []
    while (b - a) / 2 > epsilon:
        m = (a + b) / 2
        c = m - epsilon / 2
        d = m + epsilon / 2

        if f(c) < f(d):
            b = d
        else:
            a = c

        x_list.append(func((a + b) / 2))

    return (a + b) / 2, x_list

def print_info(x):
    print(f'x: {x} -> {func(x)}')
    # print(f'left: {x-0.5} -> {func(x-0.5)}')
    # print(f'right: {x+0.5} -> {func(x+0.5)}')
    print('====')

def plot_info(min_list, max_list):

    plt.plot(range(len(min_list)), min_list, 'o')
    plt.show()

    plt.plot(range(len(max_list)), max_list, 'o')
    plt.show()


if __name__ == "__main__":
    lower, upper = 2, 6
    max_iter = 1000
    tol = 1e-7

    min_point, max_point, min_list, max_list = golden_search(func=func, lower=lower, upper=upper, tol=tol)
    print(min_point, max_point)
    print(len(min_list), len(max_list))

    print_info(min_point)
    print_info(max_point)

    plot_info( min_list, max_list)

    print('\ndichotomous_search time: ')

    min_point, min_list = dichotomous_search(f=func, lower=lower, upper=upper, epsilon=tol)

    max_point, max_list = dichotomous_search(f=f_minus, lower=lower, upper=upper, epsilon=tol)

    print_info(min_point)
    print_info(max_point)

    plot_info( min_list, max_list)



