import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import root_scalar

# Parameter für die Transformation
a, b, c, d = 1.0, 2.0, 0.5, 0.3

# Transformation R = a + b * g(X)
def g(X):
    return X + (c/6)*(X**2 - 1) + (d/24)*(X**3 - 3*X) - (c**2/36)*(2*X**3 - 5*X)

# Ableitung von g(X)
def g_prime(X):
    return 1 + (c/3)*X + (d/8)*(X**2 - 1) - (c**2/6)*(X**2 - 5/6)

# Transformation R = a + b*g(X)
def R(X):
    return a + b * g(X)

# Funktion zur Berechnung der Dichte von R
def f_R(r):
    # Funktion, die die Differenz R(X) - r darstellt
    def equation(X):
        return R(X) - r
    
    # Suche nach den Wurzeln (Lösungen) für X, die die Gleichung R(X) = r erfüllen
    roots = []
    for guess in np.linspace(-10, 10, 100):
        sol = root_scalar(equation, bracket=[guess, guess + 1], method='bisect')
        if sol.converged and np.isreal(sol.root):
            root = sol.root
            if root not in roots:
                roots.append(root)
    
    # Berechnung der Dichte
    density = 0
    for root in roots:
        density += norm.pdf(root) / np.abs(b * g_prime(root))
    
    return density

# Beispiel: Zeichnen der Dichte von R
r_values = np.linspace(-10, 10, 500)
f_r_values = np.array([f_R(r) for r in r_values])

plt.plot(r_values, f_r_values, label='Dichte f_R(r)')
plt.xlabel('r')
plt.ylabel('Dichte f_R(r)')
plt.legend()
plt.show()
