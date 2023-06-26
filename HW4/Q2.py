from scipy.optimize import minimize, rosen, rosen_der
import numpy as np

M = np.array([4, 33, 31])
d = np.array([0.0298, 0.044, 0.044, 0.0138, 0.0329, 0.0329, 0.0279, 0.025, 0.025, 0.0619, 0.0317, 0.0368])
A = np.array([11.5, 92.5, 44.3, 98.1, 20.1, 6.1, 45.5, 31.0, 44.3])
d1, d2, d3a, d3k, d4, d5k, d5h, d6, d7k, d7h, d8, d9 = d
M1, M2, M3 = M

def func(F):
    return ((F / A)**2).sum()

def con_fun1(F):
    F1, F2, F3, F4, F5, F6, F7, F8, F9 = F
    return d1*F1 - d2*F2 - d3a*F3 - M1

def con_fun2(F):
    F1, F2, F3, F4, F5, F6, F7, F8, F9 = F
    return -1*d3k*F3 + d4*F4 + d5k*F5 - d6*F6 -d7k*F7 - M2

def con_fun3(F):
    F1, F2, F3, F4, F5, F6, F7, F8, F9 = F
    return d5h*F5 - d7h*F7 + d8*F8 - d9*F9 - M3

# Constraints
cons = ({'type': 'eq', 'fun': con_fun1},
        {'type': 'eq', 'fun': con_fun2},
        {'type': 'eq', 'fun': con_fun3}
)

# Bounds
bnds = ((0, None) for _ in range(9))
bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None) )

F = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
F = np.random.rand(9)

# res = minimize(func, F, method='trust-constr', tol=1e-6, bounds=bnds, constraints=cons, options={'verbose': 1})
res = minimize(func, F, method='SLSQP', tol=1e-6, bounds=bnds, constraints=cons, options={'verbose': 1})
print(all(res.x >= 0))
print(res.x)
print(res.success)
print(con_fun1(res.x), con_fun2(res.x), con_fun3(res.x))
print()
print('|$F_i$|value|')
print('|-|-|')
for i in range(len(res.x)):
    print(f'|$F_{i+1}$|{res.x[i]}|')

print('f: ', func(res.x))