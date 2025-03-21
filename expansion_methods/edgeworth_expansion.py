import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from all_methods import scipy_mvsek_to_cumulants, edgeworth_expansion

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

# Apply Edgeworth expansion
normal_expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(normal_mean, normal_var, normal_skew, normal_exkurt))
lognorm_expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt))
t_expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(t_mean, t_var, t_skew, t_exkurt))
nct_expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(nct_mean, nct_var, nct_skew, nct_exkurt))

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Edgeworth Expansion')
plt.title('Normal Distribution and Edgeworth Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Edgeworth Expansion')
plt.title('Log-Normal Distribution and Edgeworth Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Edgeworth Expansion')
plt.title('t Distribution and Edgeworth Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Edgeworth Expansion')
plt.title('Non-central t Distribution and Edgeworth Expansion')
plt.legend()

plt.tight_layout()
plt.show()
