import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)

# Function to generate Gram-Charlier expansion
def gram_charlier_expansion(x, skewness, kurtosis):
    return norm.pdf(x) * (1 + skewness/6 * hermite_polynomial(3, x) + kurtosis/24 * hermite_polynomial(4, x))

# Calculate skewness and kurtosis
normal_skew, normal_kurt = norm.stats(moments='sk')
skewed_skew, skewed_kurt = lognorm.stats(0.5, moments = 'sk')
heavy_tailed_skew, heavy_tailed_kurt = t.stats(5, moments = 'sk')

print(normal_skew, normal_kurt)
print(skewed_skew, skewed_kurt)
print(heavy_tailed_skew, heavy_tailed_kurt)

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Gram-Charlier expansion
normal_expansion = gram_charlier_expansion(x, normal_skew, normal_kurt)
skewed_expansion = gram_charlier_expansion(x, skewed_skew, skewed_kurt)
heavy_tailed_expansion = gram_charlier_expansion(x, heavy_tailed_skew, heavy_tailed_kurt)

# Plotting
plt.figure(figsize=(15, 10))

# Plot Normal distribution and its expansion
plt.subplot(3, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('Normal Distribution and Gram-Charlier Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(3, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, skewed_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('Log-Normal Distribution and Gram-Charlier Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(3, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, heavy_tailed_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('t Distribution and Gram-Charlier Expansion')
plt.legend()

plt.tight_layout()
plt.show()