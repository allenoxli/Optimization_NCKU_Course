import numpy as np
from math import sqrt
import random
import copy

def func(x):
    x1, x2 = x
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

# def golden_search(f, points, lower, upper):
#     r"""Golden-section search algorithm.
#     Find the minimum of function `f` within the interval [`a`, `b`].
#     """
#     golden_ratio = (1 + sqrt(5)) / 2

#     a, b = lower, upper
#     c = b - (b - a) / golden_ratio
#     d = a + (b - a) / golden_ratio


#     while abs(b - a) > tol:
#         if f(c) < f(d):
#             b = d
#         else:
#             a = c
#         # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
#         c = b - (b - a) / golden_ratio
#         d = a + (b - a) / golden_ratio

#     return (b + a) / 2

# def powell_conjugate(f, x1_range, x2_range, epsilon=1e-7):
#     x1, x2 = random.uniform(*x1_range), random.uniform(*x2_range)
#     h1, h2 = (1, 0), (0, 1)


#     def f1(points, n, dir_vector):
#         x = np.array((x1, x2))

#         return f()

#     golden_search(f=f, )


from numpy import identity, array, dot, zeros, argmax
from math import sqrt, ceil, log

def bracket(f, x1, h):
    c = (1 + sqrt(5)) / 2 # 1.618033989
    f1 = f(x1)
    x2 = x1 + h; f2 = f(x2)
  # Determine downhill direction and change sign of h if needed
    if f2 > f1:
        h = -h
        x2 = x1 + h; f2 = f(x2)
      # Check if minimum between x1 - h and x1 + h
        if f2 > f1: return x2,x1 - h
  # Search loop
    for i in range (100):
        h = c*h
        x3 = x2 + h; f3 = f(x3)
        if f3 > f2: return x1,x3
        x1 = x2; x2 = x3
        f1 = f2; f2 = f3
    print("Bracket did not find a mimimum")

def search(f, a, b, tol=1.0e-9):
    nIter = int(ceil(-2.078087*log(tol/abs(b-a)))) # Eq. (10.4)
    R = 0.618033989
    C = 1.0 - R
  # First telescoping
    x1 = R*a + C*b; x2 = C*a + R*b
    f1 = f(x1); f2 = f(x2)
  # Main loop
    for i in range(nIter):
        if f1 > f2:
            a = x1
            x1 = x2; f1 = f2
            x2 = C*a + R*b; f2 = f(x2)
        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = R*a + C*b; f1 = f(x1)
    if f1 < f2: return x1,f1
    else: return x2,f2

def powell(F, x, h=0.1, tol=1.0e-6):
    def f(s): return F(x + s*v)    # F in direction of v

    n = len(x)                     # Number of design variables
    df = zeros(n)                  # Decreases of F stored here
    u = identity(n)                # Vectors v stored here by rows
    for j in range(30):            # Allow for 30 cycles:
        xOld = x.copy()            # Save starting point
        fOld = F(xOld)
      # First n line searches record decreases of F
        for i in range(n):
            v = u[i]
            a, b = bracket(f, 0.0, h)
            s, fMin = search(f, a, b)
            df[i] = fOld - fMin
            fOld = fMin
            x = x + s*v
      # Last line search in the cycle
        v = x - xOld
        a, b = bracket(f, 0.0, h)
        s, fLast = search(f,a,b)
        x = x + s*v
      # Check for convergence
        if sqrt(dot(x-xOld, x-xOld) / n) < tol:
          return x, j+1
      # Identify biggest decrease & update search directions
        iMax = argmax(df)
        for i in range(iMax, n-1):
            u[i] = u[i+1]
        u[n-1] = v
    print("Powell did not converge")


def func2(x1, x2):
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

if __name__ == '__main__':

  x1_range, x2_range = (-2, 2), (-2, 2)
  x1, x2 = random.uniform(*x1_range), random.uniform(*x2_range)
  x = np.array((x1, x2))
  x_ans, iter = powell(F=func, x=x, h=0.1, tol=1.0e-6)
  print(f"iter: {iter} x: {x_ans} ans: {func(x_ans)}")
  # x0 = np.array((-2, 2))

  # minimum = optimize.fmin_powell(func2, -2, 2)
  # print(minimum)
