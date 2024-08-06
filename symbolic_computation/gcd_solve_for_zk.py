import sympy as sp

z, k = sp.symbols('z k')

eq = sp.Eq((72 - 72*z**2)/(z**6 - 3*z**4 + 9*z**2 + 9), k)

sol = sp.solve(eq, z)
print(sol)