import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

def scipy_moments_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

def cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return mean*x + 0.5*variance*x**2 + third_cumulant*x**3/6 + fourth_cumulant*x**4/24

def derivative_cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return mean + variance*x + third_cumulant*x**2/2 + fourth_cumulant*x**3/6

def second_derivative_cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return variance + third_cumulant*x + fourth_cumulant*x**2/2

def find_saddlepoint(z, mean, variance, third_cumulant, fourth_cumulant):
    """solves K'(t) = z and returns t"""
    if fourth_cumulant == 0 and third_cumulant == 0 and variance == 0:
        return None
    elif fourth_cumulant == 0 and third_cumulant == 0:
        return (z - mean)/variance
    elif fourth_cumulant == 0:
        # can this ever happen?
        sqrt_term = np.sqrt(-2*mean*third_cumulant + 2*third_cumulant*z + variance**2)
        return (sqrt_term - variance)/third_cumulant
        # return (-sqrt_term - variance)/skewness
    else:
        term_with_162 = -162*fourth_cumulant**2*mean + 162*fourth_cumulant**2*z + 162*fourth_cumulant*third_cumulant*variance - 54*third_cumulant**3
        term_with_18 = 18*fourth_cumulant*variance - 9*third_cumulant**2
        term_with_sqrt = np.sqrt(term_with_162**2 + 4*term_with_18**3)
        return 1/(3*np.cbrt(2)*fourth_cumulant) * np.cbrt(term_with_sqrt + term_with_162) - (np.cbrt(2)*term_with_18)/(3*fourth_cumulant*np.cbrt(term_with_sqrt + term_with_162)) - third_cumulant/fourth_cumulant
    
# print(find_saddlepoint(5, 0, 5/3, 0, 6))

# Function to generate Saddlepoint Approximation
def saddlepoint_approximation(x, mean, variance, third_cumulant, fourth_cumulant):
    # get saddle point for x
    s = find_saddlepoint(x, mean, variance, third_cumulant, fourth_cumulant)
    return 1/np.sqrt(2*np.pi*second_derivative_cgf(s, mean, variance, third_cumulant, fourth_cumulant)) * np.exp(cgf(s, mean, variance, third_cumulant, fourth_cumulant) - s*x)

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