import numpy as np
from scipy.optimize import differential_evolution
from sympy import symbols, lambdify

s, k = symbols('s k')

M2 = 1 + 1/(96*k**2) + (25/(1296))*s**4 - (1/36)*k*s**2
M3 = s - (76/216)*s**3 + (85/1296)*s**5 + (1/4)*k*s - (13/144)*k*s**3 + (1/32)*k**2*s
M4 = 3 + k + (7/16)*k**2 + (3/32)*k**3 + (31/3072)*k**4 - (7/216)*s**4 - (25/486)*s**6 + (21665/559872)*s**8 - (7/12)*k*s**2 + (113/452)*k*s**4 - (5155/46656)*k*s**6 - (7/24)*k**2*s**2 + (2455/20736)*k**2*s**4 - (65/1152)*k**3*s**2

a_value, b_value = 1.0, 1.0
eq1 = M3 / M2**1.5 - a_value
eq2 = M4 / M2**2 - 3 - b_value

f_eq1 = lambdify((s, k), eq1, modules='numpy')
f_eq2 = lambdify((s, k), eq2, modules='numpy')

def objective(vars):
    s_val, k_val = vars
    return np.abs(f_eq1(s_val, k_val)) + np.abs(f_eq2(s_val, k_val))

bounds = [(-10, 10), (-10, 10)]

result = differential_evolution(objective, bounds)

print(f"Found solution: s = {result.x[0]}, k = {result.x[1]}")
print(f"Error: {result.fun}")