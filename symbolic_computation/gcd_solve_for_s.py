import sympy as sp

k, s = sp.symbols('k s')

# F(k, s)
expression = k*((-s/k - (-9*k**2 - 9*s**2)/(9*k*(-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)) + (-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)/k)**4 - 6*(-s/k - (-9*k**2 - 9*s**2)/(9*k*(-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)) + (-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)/k)**2 + 3)/24 + s*((-s/k - (-9*k**2 - 9*s**2)/(9*k*(-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)) + (-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)/k)**3 + 3*s/k + (-9*k**2 - 9*s**2)/(3*k*(-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)) - 3*(-s**3 + sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4))**(1/3)/k)/6 + 1

sol = sp.solve(sp.Eq(expression, 0), s)
print(sol)