import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Definition von z(k) mit Fehlerbehandlung
def z(k):
    try:
        term1 = 2**(1/3) * (3 * np.sqrt(2) * np.sqrt(k**6 + 4*k**5 + 48*k**4 + 192*k**3) - 4*k**3)**(1/3) / k
        term2 = (18*k**2 + 216*k) / (9 * 2**(1/3) * k * (3 * np.sqrt(2) * np.sqrt(k**6 + 4*k**5 + 48*k**4 + 192*k**3) - 4*k**3)**(1/3))
        result = -np.sqrt(term1 - term2 + 1)
        if np.isnan(result) or np.isinf(result):
            return 0
        return result
    except:
        return 0

# Definition von a(z)
def a(z):
    try:
        result = (z**4 - 6*z**2 + 3) / (12*z - 4*z**3)
        if np.isnan(result) or np.isinf(result):
            return 0
        return result
    except:
        return 0

# Numerische Integration von a(z(k)) in Bezug auf k
def integrand(k):
    z_val = z(k)
    return a(z_val)

# Bereich für k
k_min, k_max = 0, 5  # Beispielbereich für k
k_values = np.linspace(k_min, k_max, 1000)

# Numerische Integration von a(z(k))
f_k_values = np.zeros_like(k_values)
for i in range(1, len(k_values)):
    f_k_values[i], _ = quad(integrand, k_min, k_values[i])

# Plotten des Ergebnisses
plt.plot(k_values, f_k_values, label='f(k)')
plt.xlabel('k')
plt.ylabel('f(k)')
plt.legend()
plt.show()
