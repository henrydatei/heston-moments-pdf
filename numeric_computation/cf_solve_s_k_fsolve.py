import numpy as np
from scipy.optimize import fsolve
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

def equations(vars):
    s_val, k_val = vars
    return [f_eq1(s_val, k_val), f_eq2(s_val, k_val)]

s_range = np.linspace(-10, 10, 10)
k_range = np.linspace(0, 10, 10)
start_points = [(s, k) for s in s_range for k in k_range]

solutions = set()

for start in start_points:
    try:
        sol = fsolve(equations, start)
        # round solution to avoid floating point errors
        sol_rounded = tuple(np.round(sol, decimals=8))
        if sol_rounded[1] >= -3 and (1 + 1/(96*sol_rounded[1]**2) + (25/(1296))*sol_rounded[0]**4 - (1/36)*sol_rounded[1]*sol_rounded[0]**2) >= 0:
            # negative kurtosis is not possible, negative variance is not possible
            solutions.add(sol_rounded)
    except Exception as e:
        print(f"no solution found for starting value {start}: {e}")

print("found solutions:")
for sol in solutions:
    print(f"s = {sol[0]}, k = {sol[1]}")

# find solution closest to the initial guess
best_sol = min(solutions, key=lambda x: np.linalg.norm(np.array(x) - np.array((a_value, b_value))))
print(f"best solution: s = {best_sol[0]}, k = {best_sol[1]}")