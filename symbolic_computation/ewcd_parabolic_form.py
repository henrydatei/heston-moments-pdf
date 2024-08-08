import sympy as sp

# Definieren Sie die Symbole
x, k, delta_x = sp.symbols('x k delta_x')

# Definieren Sie die Hermite-Polynome
He_3_x = sp.hermite_prob(3, x)
He_4_x = sp.hermite_prob(4, x)
He_6_x = sp.hermite_prob(6, x)

He_3_x_dx = sp.hermite_prob(3, x + delta_x)
He_4_x_dx = sp.hermite_prob(4, x + delta_x)
He_6_x_dx = sp.hermite_prob(6, x + delta_x)

# Definieren Sie den Ausdruck s(k, x)
s_k_x = sp.sqrt(-72 / He_6_x - 3 * k * He_4_x / He_6_x + 36 * (He_3_x / He_6_x)**2) - 6 * He_3_x / He_6_x

# Definieren Sie den Ausdruck s(k, x + delta_x)
s_k_x_dx = sp.sqrt(-72 / He_6_x_dx - 3 * k * He_4_x_dx / He_6_x_dx + 36 * (He_3_x_dx / He_6_x_dx)**2) - 6 * He_3_x_dx / He_6_x_dx

# Gleichung aufstellen und nach k lösen
equation = sp.Eq(s_k_x, s_k_x_dx)
solution = sp.solve(equation, k)

print(f"Schnittpunkt k in Abhängigkeit von x und delta_x: {solution}")
