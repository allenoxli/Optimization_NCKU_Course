import numpy as np
from math import cos, sin, sqrt
import random
import copy
import matplotlib.pyplot as plt


def func(x1, x2):
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

def golden_search(f, lower, upper, tol=1e-5):
    r"""Golden-section search algorithm.
    Find the minimum of function `f` within the interval [`a`, `b`].
    """
    golden_ratio = (1 + sqrt(5)) / 2

    a, b = lower, upper
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio


    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def euclid_dis(x1, x2, z1, z2):
    return sqrt((z1 - x1)**2 + (z2 - x2)**2)

def cyclic_coordinate(f, x1_range, x2_range, epsilon=1e-7):
    (x1_lower, x1_upper), (x2_lower, x2_upper) = x1_range, x2_range
    # Initialize.
    x1, x2 = random.uniform(*x1_range), random.uniform(*x2_range)
    prev_x1, prev_x2 = None, None

    # Save (x1, x2) each iteration for plot.
    x_list = []

    while True:
        x_list.append((x1, x2))
        prev_x1, prev_x2 = copy.deepcopy(x1), copy.deepcopy(x2)

        def f1(x):  return f(x, x2)
        def f2(x):  return f(x1, x)

        min_x1 = golden_search(f=f1, lower=x1_lower, upper=x1_upper, tol=epsilon)
        min_x2 = golden_search(f=f2, lower=x2_lower, upper=x2_upper, tol=epsilon)

        scores = (f(min_x1, x2), f(x1, min_x2))
        idx = np.argmin(scores)

        if idx == 0:
            x1 = min_x1
        if idx == 1:
            x2 = min_x2

        if euclid_dis(prev_x1, prev_x2, x1, x2) < epsilon:
            break

    return x1, x2, x_list

def plot_point(x_list):
    x1_list = [x[0] for x in x_list]
    x2_list = [x[1] for x in x_list]

    plt.plot(x1_list, x2_list)
    plt.show()


if __name__ == '__main__':
    # print(random.uniform(*a))
    print(np.argmax((5,7,3,1)))

    x1, x2, x_list = cyclic_coordinate(
        f=func,
        x1_range=(-2, 2),
        x2_range=(-2, 2),
        epsilon=1e-7
    )

    print(f"(x1, x2) = ({x1}, {x2}) minimum is {func(x1, x2)}")
    plot_point(x_list)