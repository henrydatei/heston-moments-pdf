import sympy as sp

s, k = sp.symbols('s k')
a_value, b_value = 1.0, 1.0

M2 = 1 + 1/(96*k**2) + (25/(1296))*s**4 - (1/36)*k*s**2
M3 = s - (76/216)*s**3 + (85/1296)*s**5 + (1/4)*k*s - (13/144)*k*s**3 + (1/32)*k**2*s
M4 = 3 + k + (7/16)*k**2 + (3/32)*k**3 + (31/3072)*k**4 - (7/216)*s**4 - (25/486)*s**6 + (21665/559872)*s**8 - (7/12)*k*s**2 + (113/452)*k*s**4 - (5155/46656)*k*s**6 - (7/24)*k**2*s**2 + (2455/20736)*k**2*s**4 - (65/1152)*k**3*s**2

eq1 = sp.Eq(M3 / M2**1.5, a_value)
eq2 = sp.Eq(M4 / M2**2 - 3, b_value)

initial_guesses = [(0.1, 0.1), (0.1, 1), (1, 0.1), (1, 1)]

solutions = []
for guess in initial_guesses:
    try:
        sol = sp.nsolve([eq1, eq2], [s, k], guess)
        if sol not in solutions: 
            solutions.append(sol)
    except Exception as e:
        print(f"No solution found for starting value {guess}: {e}")

for i, sol in enumerate(solutions, 1):
    print(f"Solution {i}: s = {sol[0]}, k = {sol[1]}")
