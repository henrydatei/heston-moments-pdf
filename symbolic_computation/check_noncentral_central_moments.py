# Check noncentral and central moments formulas from Okhrin et al 2022

import sympy as sp

# Define the symbols
mu, theta, kappa, rho, sigma, t = sp.symbols('mu theta kappa rho sigma t')

# Define mu_1 and mu_2
mu_1 = (mu - theta / 2) * t
mu_2 = (1 / (4 * kappa**3)) * (
    sp.exp(-kappa * t) * (
        sp.exp(kappa * t) * (
            kappa**3 * t * (t * (theta - 2 * mu)**2 + 4 * theta)
            - 4 * kappa**2 * rho * sigma * t * theta
            + kappa * sigma * theta * (4 * rho + sigma * t)
            - sigma**2 * theta
        )
        + sigma * theta * (sigma - 4 * kappa * rho)
    )
)

# Calculate the variance
variance = mu_2 - mu_1**2

# Define the expression to check against
expression_to_check = theta * (
    -4 * kappa**2 * rho * sigma * t
    + 4 * kappa**3 * t
    + sigma * sp.exp(-kappa * t) * (sigma - 4 * kappa * rho)
    + 4 * kappa * sigma * rho
    + kappa * sigma**2 * t
    - sigma**2
) / (4 * kappa**3)

# Check equivalence
are_equivalent_final_check = sp.simplify(variance - expression_to_check) == 0

# Print result
print("Are the expressions equivalent?", are_equivalent_final_check)
