import sympy as sp

k, a, b, c, d, e, f = sp.symbols('k a b c d e f')

s1 = sp.sqrt(a + b * k) + c
s2 = sp.sqrt(d + e * k) + f

equation = sp.Eq(s1, s2)

solution_k = sp.solve(equation, k)

solution_y = [s1.subs(k, k_val) for k_val in solution_k]

for i, (k_val, y_val) in enumerate(zip(solution_k, solution_y), start=1):
    print(f"Intersection {i}:")
    print(f"  k = {k_val}")
    print(f"  s = {y_val}")
