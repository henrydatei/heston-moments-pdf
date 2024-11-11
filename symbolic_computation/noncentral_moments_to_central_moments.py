import sympy as sp

# Define new symbols for the moments and parameters
mu, theta, kappa, sigma = sp.symbols('mu theta kappa sigma')

# Define the non-central moments
mu_1 = 1 + mu
mu_2 = (mu + 1)**2 + theta
mu_3 = (mu + 1)**3 + 3*theta + 3*mu*theta
mu_4 = (1 / (kappa * (kappa - 2))) * (
    kappa**2 * mu**4 + 4 * kappa**2 * mu**3 + 6 * kappa**2 * mu**2 * theta
    - 2 * kappa * mu**4 + 6 * kappa**2 * mu**2 + 12 * kappa**2 * mu * theta
    + 3 * kappa**2 * theta**2 - 8 * kappa * mu**3 - 12 * kappa * mu**2 * theta
    + 4 * kappa**2 * mu + 6 * kappa**2 * theta - 12 * kappa * mu**2
    - 24 * kappa * mu * theta - 6 * kappa * theta**2 - 3 * sigma**2 * theta
    + kappa**2 - 8 * kappa * mu - 12 * kappa * theta - 2 * kappa
)

# Calculate central moments
mean = mu_1 # First central moment (mean)
variance = mu_2 - mu_1**2 # Second central moment (variance)
skewness = (mu_3 - 3 * mu_1 * mu_2 + 2 * mu_1**3) / (variance**(3/2)) # skewness
kurtosis = (mu_4 - 4 * mu_1 * mu_3 + 6 * mu_1**2 * mu_2 - 3 * mu_1**4) / (variance**2) # kurtosis

# Simplify the results
variance = variance.simplify()
skewness = skewness.simplify()
kurtosis = kurtosis.simplify()

# Display results
print(mean)
print(variance)
print(skewness)
print(kurtosis)
