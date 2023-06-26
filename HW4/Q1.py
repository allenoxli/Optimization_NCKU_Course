from scipy.optimize import minimize, rosen, rosen_der
from math import pi
import numpy as np 

def func(args):
    r, h = args[0], args[1]
    return 2*pi*r*(r + h) + (pi * r**2 * h - 20)**2 

def volume(r, h):
    return pi * r**2 * h

# Constraints
cons = ({'type': 'eq', 'fun': lambda x:  pi * x[0]**2 * x[1] - 20},
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]}
)

# Bounds
bnds = ((0, None), (0, None))
x0 = np.array([1, 1])
# res = minimize(func, x0, method='BFGS', tol=1e-6, bounds=bnds, constraints=cons)
res = minimize(func, x0, method='SLSQP', tol=1e-6, bounds=bnds, constraints=cons)
# res = minimize(func, x0, method='trust-constr', tol=1e-6, bounds=bnds, constraints=cons)
print('Converge: ', res.success)
print('(r, h): ' , res.x)
r, h = res.x
print('penalty part: ', (pi * r**2 * h - 20)**2)
print('Surface Area: ', func(res.x) - (pi * r**2 * h - 20)**2)
print('Volume: ', volume(r, h))