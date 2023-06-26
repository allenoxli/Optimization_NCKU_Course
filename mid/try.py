import numpy as np
from sympy import diff, symbols, sin, tan
import math

def func(p):
    x, y = symbols('x,y')
    f = 5*x**6 + 18*x**5 -30*x**4 -120*x**3 + 30*x*y**2
    df_x = diff(f, x)
    df_y = diff(f, y)

    print('df: ')
    print(df_x)
    print(df_y)
    print('----')

    df_x2 = diff(df_x, x)
    df_x2_val = df_x2.subs({x:-2, y:0})
    print(f'df_x2: {df_x2}')
    print(f'df_x2_val: {df_x2_val}')
    print('----')


    p_dict = {x:p[0], y:p[1]}
    df_val = np.array([df_x.subs(p_dict), df_y.subs(p_dict)])
    print(f'df_val: {df_val}')

    s = -1 * df_val
    print(f's:{s}')

    Armijo(p, s, f)

def Armijo(p, s, f, epsilon=0.2):
    l, x, y = symbols('l,x,y')
    p_new = p + l*s
    print('Amijro-------')
    print(p_new)
    p_dict = {x:p_new[0], y:p_new[1]}
    f_lambda = f.subs(p_dict)

    # First order of derivative of `f_lambda`. That is f_lambda'.
    df_lambda = diff(f_lambda, l)
    # f_lambda(0) value.
    f_zero = f_lambda.subs({l:0})
    # f_lambda'(0) value.
    df_zero = df_lambda.subs({l:0})

    # We define F(lambda) = f(0) + epsilon*lambda*f'(0), here f stand for `f_lambda`.
    F = f_zero + df_zero*epsilon*l

    print(f'f_lambda: {f_lambda}')
    print(f'df_lambda: {df_lambda}')
    print(f'f_zero: {f_zero}')
    print(f'df_zero: {df_zero}')

    # Arbitrary initialize lambda's value.
    lambda_val = 1.0
    iter = 0
    while True:
        f_value = f_lambda.subs({l:lambda_val})
        F_value = F.subs({l:lambda_val})
        print(f'lambda_val: {lambda_val}')
        print(f'f, F: {f_value}, {F_value}')
        if f_value >= F_value:
            lambda_val /= 2
        else:
            return lambda_val
            lambda_val *= 2
        iter += 1
        # if iter >= 10:
        #     break

    # print(lambda_val)


# p = np.array([1, 1])
# func(p)

# 9900(1 - 60l)^2 + 9900(330l + 1)^5 + 29700(330l + 1)^4 - 39600(330l + 1)^3 - 118800(330l + 1)^2 + (330l + 1)(216000l - 3600)


# x, y, z, l = symbols('x,y,z,l')
# f = 5*x**6 + 18*x**5 -30*x**4 -120*x**3 + 30*x*y**2
# df_x = diff(f, x)
# df_y = diff(f, y)

# print('df: ')
# print(df_x)
# print(df_y)
# print('----')

# df_x2 = diff(df_x, x)
# df_x2_val = df_x2.subs({x:-2, y:0})
# print(f'df_x2: {df_x2}')
# print(f'df_x2_val: {df_x2_val}')
# print('----')

# p = np.array([29/4, 7/4, 4/3])
# p_dict = {x:p[0], y:p[1], z:p[2]}
# g1 = -8*x - z - 7
# g2 = -2*x + 2*y + 11
# g3 = z - x - 2*y - 13

# print(f'g1: {g1.subs(p_dict)}')
# print(f'g2: {g2.subs(p_dict)}')
# print(f'g3: {g3.subs(p_dict)}')
# print('====')
# p_new = p + l*np.array([1/4, 1/4, -1/6])
# p_dict = {x:p_new[0], y:p_new[1], z:p_new[2]}
# print(p_new)
# print(f'g1: {g1.subs(p_dict)}')
# print(f'g2: {g2.subs(p_dict)}')
# print(f'g3: {g3.subs(p_dict)}')
# print('====')


# 4x-18-5a-2b-c=0,8y-25+2b-2c=0,6z-8-a+c=0,
# -5x-z-7=0,-2x+2y+11=0, z-x-2y-13=0


# Q3
x, y, z, l = symbols('x,y,z,l')
f = x / (1 + x**2)
p_dict= {x:0}

df_x = diff(f, x)
df_val = df_x.subs(p_dict)

df_x = diff(df_x, x)
ddf_val = df_x.subs(p_dict)



print('df: ')
print(df_val)
print(ddf_val)
print('----')


# -2x^2/(x^2 + 1)^2 + 1/(x^2 + 1) = 0
8*x**3/(x**2 + 1)**3 - 6*x/(x**2 + 1)**2