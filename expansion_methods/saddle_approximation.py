import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

def scipy_moments_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

def cgf(x, mean, variance, skewness, kurtosis):
    return mean*x + 0.5*variance*x**2 + skewness*x**3/6 + kurtosis*x**4/24

def derivative_cgf(x, mean, variance, skewness, kurtosis):
    return mean + variance*x + skewness*x**2/2 + kurtosis*x**3/6

def second_derivative_cgf(x, mean, variance, skewness, kurtosis):
    return variance + skewness*x + kurtosis*x**2/2

def find_saddlepoint(z, mean, variance, skewness, kurtosis):
    """solves K'(t) = z and returns t"""
    if kurtosis == 0 and skewness == 0 and variance == 0:
        return None
    elif kurtosis == 0 and skewness == 0:
        return (z - mean)/variance
    elif kurtosis == 0:
        # can this ever happen?
        sqrt_term = np.sqrt(-2*mean*skewness + 2*skewness*z + variance**2)
        return (sqrt_term - variance)/skewness
        # return (-sqrt_term - variance)/skewness
    else:
        term_with_162 = -162*kurtosis**2*mean + 162*kurtosis**2*z + 162*kurtosis*skewness*variance - 54*skewness**3
        term_with_18 = 18*kurtosis*variance - 9*skewness**2
        term_with_sqrt = np.sqrt(term_with_162**2 + 4*term_with_18**3)
        return 1/(3*np.cbrt(2)*kurtosis) * np.cbrt(term_with_sqrt + term_with_162) - (np.cbrt(2)*term_with_18)/(3*kurtosis*np.cbrt(term_with_sqrt + term_with_162)) - skewness/kurtosis
    
# print(find_saddlepoint(5, 0, 5/3, 0, 6))

# Function to generate Saddlepoint Approximation
def saddlepoint_approximation(x, mean, variance, skewness, kurtosis):
    # get saddle point for x
    s = find_saddlepoint(x, mean, variance, skewness, kurtosis)
    return 1/np.sqrt(2*np.pi*second_derivative_cgf(s, mean, variance, skewness, kurtosis)) * np.exp(cgf(s, mean, variance, skewness, kurtosis) - s*x)

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

# Apply Saddlepoint Approximation
normal_approximation = saddlepoint_approximation(x, *scipy_moments_to_cumulants(normal_mean, normal_var, normal_skew, normal_exkurt))
lognorm_approximation = saddlepoint_approximation(x, *scipy_moments_to_cumulants(lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt))
t_approximation = saddlepoint_approximation(x, *scipy_moments_to_cumulants(t_mean, t_var, t_skew, t_exkurt))
nct_approximation = saddlepoint_approximation(x, *scipy_moments_to_cumulants(nct_mean, nct_var, nct_skew, nct_exkurt))

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its approximation
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_approximation, 'b-', label='Saddlepoint Approximation')
plt.title('Normal Distribution and Saddlepoint Approximation')
plt.legend()

# Plot Skewed distribution and its approximation
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_approximation, 'b-', label='Saddlepoint Approximation')
plt.title('Log-Normal Distribution and Saddlepoint Approximation')
plt.legend()

# Plot Heavy-tailed distribution and its approximation
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_approximation, 'b-', label='Saddlepoint Approximation')
plt.title('t Distribution and Saddlepoint Approximation')
plt.legend()

# Plot Non-central t distribution and its approximation
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_approximation, 'b-', label='Saddlepoint Approximation')
plt.title('Non-central t Distribution and Saddlepoint Approximation')
plt.legend()

plt.tight_layout()
plt.show()