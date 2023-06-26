# In[]
r""""Linear equation y = mx + b.
Using Armijo to find lambda to update x. (ex: x2 = x1 + \lambda * S1)
Using Fletcher-Reeves and Quasi-Newton method to update s. (s2 = -f'(x2) + \beta1 * S1)
(\beta1 f'(x2))
"""
import numpy as np
from sympy import diff, symbols, sin, tan
import math

x = np.array([0.4, 1, 1.5, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
y = np.array([28, 31, 30, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32])

def Fletcher_Reeves(x, y, p, degree=1, tol=5e-1, max_iter=100):
    r"""Linear equation y = mx + b.
    To find m and b, we can minimize f(m, b) = \sum (mx + b - y)**2.
    """
    m, b = symbols('m,b')
    # Linear equation f(m, b) = \sum (mx + b - y)**2.
    f = ((m*x + b - y)**2).sum()

    # First order of derivative of f(m, b).
    df = [diff(f, m), diff(f, b)]

    for step in range(max_iter):

        p_dict = {m:p[0], b:p[1]}
        # \delta f(m, b)
        df_val = np.array([df[0].subs(p_dict), df[1].subs(p_dict)])

        # Direction vector. s = -\delta f.
        s = -1 * df_val

        if math.sqrt((s**2).sum()) < tol:
            print('tol')
            break

        # Find proper lambda value.
        lambda_val = Armijo(p, s, f)

        # Update p = (m , b) to get a new point p_{j+1} = p_j + lambda * s_j.
        p = p + lambda_val * s

        p_dict = {m:p[0], b:p[1]}
        df_new_val = np.array([df[0].subs(p_dict), df[1].subs(p_dict)])

        beta = (df_new_val.T @ df_new_val) / (df_val.T @ df_val)

        # Update direction vector `s`. s_{j+1} = -\delta f_j + \beta * s_j.
        s = -1*df_new_val + beta*s

        if step % 5 == 0:
            print()
            print(f'step: {step}')
            print(f'p: {p}')

    print('stop step: ', step)

    return p

def Armijo(p, s, f, epsilon=0.2, max_iter=1):
    r"""s: direction vector."""

    m,b,l = symbols('m,b,l') # `l` for lambda
    p_new = p + l*s
    p_dict = {m:p_new[0], b:p_new[1]}
    f_lambda = f.subs(p_dict)
    # First order of derivative of `f_lambda`. That is f_lambda'.
    df_lambda = diff(f_lambda, l)
    # f_lambda(0) value.
    f_zero = f_lambda.subs({l:0})
    # f_lambda'(0) value.
    df_zero = df_lambda.subs({l:0})

    # We define F(lambda) = f(0) + epsilon*lambda*f'(0), here f stand for `f_lambda`.
    F = f_zero + df_zero*epsilon*l

    # Arbitrary initialize lambda's value.
    lambda_val = 1.0
    iter = 0
    while True:
        f_value = f_lambda.subs({l:lambda_val})
        F_value = F.subs({l:lambda_val})
        if f_value >= F_value:
            lambda_val /= 2
        else:
            # if iter >= max_iter:
            return lambda_val
            lambda_val *= 2
        iter += 1

def quasi_newton(x, y, p, degree=1, tol=5e-1, max_iter=100):
    m, b = symbols('m,b')
    # Linear equation f(m, b) = (mx + b - y)**2.
    f = ((m*x + b - y)**2).sum()

    # First order of derivative of f(m, b).
    df = [diff(f, m), diff(f, b)]

    beta = np.identity(degree+1)

    for step in range(max_iter):
        p_dict = {m:p[0], b:p[1]}
        df_val = np.array([df[0].subs(p_dict), df[1].subs(p_dict)])

        s = -1 * beta @ df_val

        if math.sqrt((s**2).sum()) < tol:
            print('tol')
            break

        # Find proper lambda value.
        lambda_val = Armijo(p, s, f)

        # Update p = (m , b).
        p_new = p + lambda_val * s

        p_dict = {m:p_new[0], b:p_new[1]}
        df_new_val = np.array([df[0].subs(p_dict), df[1].subs(p_dict)])

        d = p_new - p
        g = df_new_val - df_val

        d, g = d[:,None], g[:,None]

        beta = beta + (d @ d.T)/(d.T @ g) - (beta @ g @ g.T @ beta)/(g.T @ beta @ g)

        s = -1 * beta @ df_new_val

        p = p_new

        if step % 5 == 0:
            print(f'step: {step}')
            print(f'p: {p}')

    print(f'stop step: ', step)

    return p


if __name__ == '__main__':
    # Initialize (m, b).
    p = np.array([1.0, 1.0])
    # p = np.array([0.49432466, 29.5370108])
    p = Fletcher_Reeves(x, y, p, degree=1, max_iter=10)
    print('Fletcher_Reeves final ans: ', p)
    print('========')
    p = quasi_newton(x, y, p, degree=1, tol=5e-1, max_iter=400)
    print('quasi_newton final ans: ', p)


# %%
