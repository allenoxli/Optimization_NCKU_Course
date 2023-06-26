import numpy as np
from math import sqrt
import random
import copy
import matplotlib.pyplot as plt

def func(x):
    x1, x2 = x
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

def f_new(x, s, v):
    return func(x + s*v)

def gold_search(f, x, lower, upper, v, tol=1e-5):
    golden_ratio = (1 + sqrt(5)) / 2
    a, b = copy.deepcopy(lower), copy.deepcopy(upper)
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(b - a) > tol:
        if f(x, c, v) < f(x, d, v):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def check_norm(v):
    L2_norm = (v**2).sum()
    if L2_norm > 1:
        v /= L2_norm
    return v

def euclid_dis(x, x_new):
    z1, z2 = x
    x1, x2 = x_new
    return sqrt((z1 - x1)**2 + (z2 - x2)**2)

def check_x(x):
    pass

def powell(f, x, lower, upper, tol=1e-5):
    n = len(x)
    f_table = np.zeros(n)
    s_table = np.zeros(n)
    # 方向向量 u = (v1,...vn)
    u = np.identity(n)
    # v1 = np.array((random.uniform(-1, 1), random.uniform(-1, 1)))
    # v2 = np.array((random.uniform(-1, 1), random.uniform(-1, 1)))
    # v1, v2 = check_norm(v1), check_norm(v2)
    # u[0], u[1] = v1, v2

    max_iter = 100

    # Save (x1, x2) each iteration for plot.
    x_list = []

    for iter in range(max_iter):
        x_list.append(x)
        x_old = x.copy()
        for i in range(n):
            v = u[i]    # 方向向量 u[i] = vi
            lower_new, upper_new = lower-x[i], upper-x[i]
            s = gold_search(f=f_new, x=x, lower=lower_new, upper=upper_new, v=v, tol=tol)
            s_table[i] = s
            x_new = x + s*v
            f_table[i] = f(x_new)

        v_new = x_new - x
        check_norm(v_new)
        min_idx = np.argmin(f_table)
        print(f'min_idx: {min_idx}')
        # if sqrt(((x_new-x)**2).sum()) < tol:
        #     return x_new, x_list

        if euclid_dis(x, x_new) < tol:
            return x_new, x_list

        x = x + s_table[min_idx]*u[min_idx]
        for j in range(0, n-1):
            u[j] = u[j+1]
        u[n-1] = v_new

    print("no convergence\n")

def plot_point(x_list):
    x1_list = [x[0] for x in x_list]
    x2_list = [x[1] for x in x_list]

    plt.plot(x1_list, x2_list)
    plt.show()

if __name__ == '__main__':

    x1_range, x2_range = (-2, 2), (-2, 2)
    lower, upper = -2, 2
    # Start point x = (x1, x2).
    x1, x2 = random.uniform(*x1_range), random.uniform(*x2_range)
    # x1, x2 = 0, -2

    print(f'start point(x1, x2) = ({x1}, {x2})\n')
    x_start = np.array((x1, x2))

    x_ans, x_list = powell(f=func, x=x_start, lower=lower, upper=upper, tol=1e-6)
    print(f"x: {x_ans} ans: {func(x_ans)}")
    plot_point(x_list)

