import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)

# plot hermite polynomials
# x = np.linspace(-4, 6, 1000)
# for i in range(6):
#     plt.plot(x, [hermite_polynomial(i, xi) for xi in x], label=f'Hermite {i}')
# plt.legend()
# plt.ylim(-10, 20)
# plt.show()

def scipy_moments_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

# Function to generate Gram-Charlier expansion
def gram_charlier_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/6 * hermite_polynomial(3, z) + fourth_cumulant/24 * hermite_polynomial(4, z))

# Calculate skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_exkurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_exkurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_exkurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_skew, normal_exkurt)
print(lognorm_skew, lognorm_exkurt)
print(t_skew, t_exkurt)
print(nct_skew, nct_exkurt)

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Gram-Charlier expansion
normal_expansion = gram_charlier_expansion(x, *scipy_moments_to_cumulants(normal_mean, normal_var, normal_skew, normal_exkurt))
lognorm_expansion = gram_charlier_expansion(x, *scipy_moments_to_cumulants(lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt))
t_expansion = gram_charlier_expansion(x, *scipy_moments_to_cumulants(t_mean, t_var, t_skew, t_exkurt))
nct_expansion = gram_charlier_expansion(x, *scipy_moments_to_cumulants(nct_mean, nct_var, nct_skew, nct_exkurt))

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('Normal Distribution and Gram-Charlier Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('Log-Normal Distribution and Gram-Charlier Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('t Distribution and Gram-Charlier Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Gram-Charlier Expansion')
plt.title('Non-central t Distribution and Gram-Charlier Expansion')
plt.legend()

plt.tight_layout()
plt.show()
