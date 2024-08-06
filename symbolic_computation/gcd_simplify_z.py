import sympy as sp

z, k, s = sp.symbols('z k s')

# one real solution for z from WolframAlpha
z_expr = (sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4) - s**3)**(1/3)/k - (-9*k**2 - 9*s**2)/(9*k * (sp.sqrt(-k**6 - 3*k**4*s**2 - 3*k**2*s**4) - s**3)**(1/3)) - s/k

# F(k, s, z)
F = 1 + s/6*(z**3 - 3*z) + k/24*(z**4 - 6*z**2 + 3)

# substitute z in F(k, s, z) with z_expr
F_substituted = F.subs(z, z_expr)

# simplify the expression
simplified_F = sp.simplify(F_substituted)

print(F_substituted)