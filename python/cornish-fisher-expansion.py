import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

def cornish_fisher_expansion(x, mean, var, skew, kurt):
    """Returns the Cornish-Fisher expansion of a random variable x with skewness skew and kurtosis kurt.
    Source: "Option Pricing Under Skewness and Kurtosis Using a Cornish-Fisher Expansion"

    Args:
        x (float): The value of the random variable.
        mean (float): The mean of the random variable.
        var (float): The variance of the random variable.
        skew (float): The skewness of the random variable.
        kurt (float): The kurtosis of the random variable.
    """
    x = (x - mean) / np.sqrt(var)
    a = (-skew)/(kurt/8 - (skew**2)/3)
    b = kurt/24 - (skew**2)/18
    p = (1 - kurt/8 + 5*(skew**2)/36)/(kurt/24 - (skew**2)/18) - 1/3 * ((skew**2)/36)/((kurt/24 - (skew**2)/18)**2)
    q = (-skew)/(kurt/8 - (skew**2)/3) - 1/18 * (skew * (1 - kurt/8 + 5*skew**2/36))/((kurt/24 - skew**2/18)**2) - 2/27 * (skew**3/216)/((kurt/24 - skew**2/18)**3)
    z = a/3 + np.cbrt((-q + x/b + np.sqrt((q-x/b)**2 + 4/27*p**3))/2) + np.cbrt((-q + x/b - np.sqrt((q-x/b)**2 + 4/27*p**3))/2)
    # print(f'{a=}')
    # print(f'{b=}')
    # print(f'{p=}')
    # print(f'{q=}')
    # print(f'{z=}')
    # print((q-x/b)**2 + 4/27*p**3)
    return 1/np.sqrt(2*np.pi) * np.exp(-z**2/2)/(z**2 * (kurt/8 - skew**2/6) + z*skew/3 + 1 - kurt/8 + 5*skew**2/36)

# print(cornish_fisher_expansion(0.5, 0.5, 2))

# Calculate skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_kurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_kurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_kurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_kurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_skew, normal_kurt)
print(lognorm_skew, lognorm_kurt)
print(t_skew, t_kurt)
print(nct_skew, nct_kurt)

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Gram-Charlier expansion
normal_expansion = cornish_fisher_expansion(x, normal_mean, normal_var, normal_skew, normal_kurt)
lognorm_expansion = cornish_fisher_expansion(x, lognorm_mean, lognorm_var,lognorm_skew, lognorm_kurt)
t_expansion = cornish_fisher_expansion(x, t_mean, t_var, t_skew, t_kurt)
nct_expansion = cornish_fisher_expansion(x, nct_mean, nct_var, nct_skew, nct_kurt)

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Normal Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Log-Normal Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('t Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Non-central t Distribution and Cornish-Fisher Expansion')
plt.legend()

plt.tight_layout()
plt.show()
