import sympy as sp

# Definiere die Symbole
z, x, y = sp.symbols('z x y')

# Definiere den Ausdruck für z
z_expr = (sp.sqrt(-x**6 - 3*x**4*y**2 - 3*x**2*y**4) - y**3)**(1/3)/x - (-9*x**2 - 9*y**2)/(9*x * (sp.sqrt(-x**6 - 3*x**4*y**2 - 3*x**2*y**4) - y**3)**(1/3)) - y/x

# Definiere den ursprünglichen Ausdruck
expr = ((z**4 - 6*z**2 + 3)/(12*z - 4*z**3))*x + 24/(12*z - 4*z**3)

# Substituiere z durch z_expr
expr_substituted = expr.subs(z, z_expr)

# Vereinfache den Ausdruck
simplified_expr = sp.simplify(expr_substituted)

# Ausgabe des vereinfachten Ausdrucks
# print(simplified_expr)

sol = sp.solve(sp.Eq(simplified_expr, 0), y)
print(sol)