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
mu_3 = (sp.exp(-kappa * t))/(8*kappa**5) * (
    sp.exp(kappa*t) * (
        3 * kappa**2 * sigma**2 * theta * (
            -2*mu*t + 16*rho**2 + 6*rho*sigma*t + t*sigma + 4
        ) 
        - 3 * kappa**3 * sigma * theta * (
            rho*(-8*mu*t + 4*t*theta + 8) + sigma*t*(-2*mu*t + theta*t + 4) + 8*rho**2*sigma*t
        )
        + 12 * kappa**4 * rho * sigma * t * theta * (
            -2*mu*t + theta*t + 2
        )
        + kappa**5 * t**2 * (2*mu-theta) * (t*(theta-2*mu)**2 + 12*theta) 
        - 3 * kappa * sigma**3 * theta * (12*rho+sigma*t) 
        + 6 * sigma**4 * theta
    ) 
    - 3 * sigma * theta * (
        kappa**2 * sigma * (
            -2*mu*t + 16*rho**2 - 6*rho*sigma*t + theta*t + 4
        ) 
        + 4 * kappa**3 * rho * (
            2*mu*t + 2*rho*sigma*t - theta*t - 2
        ) 
        + kappa * sigma**2 * (sigma*t - 12*rho) 
        + 2 * sigma**3
    )
)
mu_4 = (1 / (32 * kappa**7)) * (
    2 * kappa * t * (
        6 * kappa**4 * t**2 * theta * (theta - 2 * mu)**2 * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)
        - 12 * kappa**2 * sigma * t * theta * (theta - 2 * mu) * (2 * kappa * rho - sigma) * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)
        + kappa**6 * t**3 * (theta - 2 * mu)**4
        + 3 * sigma**2 * theta * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2) * (4 * kappa**2 * (4 * rho**2 + 1) - 20 * kappa * rho * sigma + 5 * sigma**2)
        + 3 * kappa**2 * t * theta**2 * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)**2
    )
    - 24 * kappa**2 * sigma * t * theta * sp.exp(-kappa * t) * (2 * mu - theta) * (2 * kappa * rho - sigma) * (
        4 * kappa**2 * (sp.exp(kappa * t) + rho * sigma * t - 1)
        - kappa * sigma * (8 * rho * (sp.exp(kappa * t) - 1) + sigma * t)
        + 2 * sigma**2 * (sp.exp(kappa * t) - 1)
    )
    + 12 * kappa**2 * sigma * t * theta * sp.exp(-kappa * t) * (sp.exp(kappa * t) - 1) * (4 * kappa * rho - sigma) * (
        kappa**2 * (t * (theta - 2 * mu)**2 + 4 * theta)
        - 4 * kappa * rho * sigma * theta
        + sigma**2 * theta
    )
    + 3 * sigma**2 * theta * sp.exp(-2 * kappa * t) * (
        -4 * sp.exp(kappa * t) * (
            4 * kappa**3 * (
                sigma * (
                    -3 * (8 * rho**2 + 1) * sigma * t
                    + 24 * (rho**3 + rho)
                    + 2 * rho * sigma**2 * t**2
                ) + 4 * rho**2 * theta
            )
            - kappa**2 * sigma * (sigma * (136 * rho**2 - 40 * rho * sigma * t + sigma**2 * t**2 + 24) + 8 * rho * theta)
            + 16 * kappa**5 * rho**2 * t * (rho * sigma * t - 2)
            + 4 * kappa**4 * (rho * (rho * (
                sigma * t * (16 * rho - 5 * sigma * t) - 16) + 12 * sigma * t) - 2)
            + kappa * sigma**2 * (56 * rho * sigma - 5 * sigma**2 * t + theta)
            - 7 * sigma**4
        ) + sp.exp(2 * kappa * t) * (
            32 * kappa**3 * rho * (12 * (rho**2 + 1) * sigma + rho * theta)
            - 16 * kappa**2 * sigma * (35 * rho**2 * sigma + rho * theta + 6 * sigma)
            - 32 * kappa**4 * (8 * rho**2 + 1)
            + 2 * kappa * sigma**2 * (116 * rho * sigma + theta)
            - 29 * sigma**4
        ) + (sigma - 4 * kappa * rho)**2 * (2 * kappa * theta + sigma**2)
    )
)

# Calculate the zetas
zeta_2_check = mu_2 - mu_1**2
zeta_3_check = (mu_3 - 3 * mu_1 * mu_2 + 2 * mu_1**3) / (zeta_2_check**(3/2))
zeta_4_check = (mu_4 - 4 * mu_1 * mu_3 + 6 * mu_1**2 * mu_2 - 3 * mu_1**4) / (zeta_2_check**2)

# Define the expression to check against
zeta_2 = theta * (
    -4 * kappa**2 * rho * sigma * t
    + 4 * kappa**3 * t
    + sigma * sp.exp(-kappa * t) * (sigma - 4 * kappa * rho)
    + 4 * kappa * sigma * rho
    + kappa * sigma**2 * t
    - sigma**2
) / (4 * kappa**3)
zeta_3 = (
    3 * kappa * sigma * theta * sp.exp(kappa * t / 2) * (sigma - 2 * kappa * rho)
) / (
    zeta_2**(3/2)
) * (
    4 * kappa**2 * (
        sp.exp(kappa * t) * (rho * sigma * t + 1) + rho * sigma * t - 1
    ) 
    - 4 * kappa**3 * t * sp.exp(kappa * t) 
    - kappa * sigma * (
        sp.exp(kappa * t) * (8 * rho + sigma * t) - 8 * rho + sigma * t
    ) 
    + 2 * sigma**2 * (sp.exp(kappa * t) - 1)
)
zeta_4 = (3*sp.exp(-2*kappa*t))/(zeta_2**2) * (
    4 * sigma * sp.exp(kappa*t) * (
        4 * kappa**4 * sigma * (
            rho**2 * (
                5*sigma**2*t**2 + 4*t*sigma + 16
            )
            - 16*rho**3*sigma*t
            - 12*rho*sigma*t
            + sigma*t + 2
        )
        - 4 * kappa**3 * sigma * (
            4 * rho**2 * (
                theta - 6*sigma**2*t
            )
            + 24 * rho**3 * sigma
            + 2 * rho * sigma * (
                sigma**2*t**2 + t*theta + 12
            )
            - 3 * sigma**2 * t
        )
        + kappa**2 * sigma**2 * (
            sigma * (
                136*rho**2 + t*theta + 24
            )
            - 40*rho*sigma**2*t + 8*rho*theta + sigma**3*t**2 
        )
        - 16 * kappa**5 * rho * t * (
            rho**2 * sigma**2 * t - 2 * rho * sigma + theta
        )
        - kappa * sigma**3 * (
            56*rho*sigma - 5*sigma**2*t + theta
        )
        + 7 * sigma**5
    )
    + sp.exp(2*kappa*t) * (
        -16 * kappa**4 * sigma**2 * (
            8 * rho**3 * sigma * t + 4 * rho**2 * (t*theta + 4) + rho * sigma * t * (t*theta + 12) + t * theta + 2
        )
        + 2 * kappa**3 * sigma**2 * (
            16 * rho**2 * (6*sigma**2*t + theta) + 192 * rho**3 * sigma + 16 * rho * sigma * (t*theta + 12) + sigma**2 * t * (t*theta + 24)
        )
        - 4 * kappa**2 * sigma**3 * (
            sigma * (
                140*rho**2 + t*theta + 24
            )
            + 20 * rho * sigma**2 * t + 4 * rho * theta
        )
        + 16 * kappa**5 * sigma * t * (
            sigma * (
                2 * rho**2 * (t*theta+4) + t*theta + 2
            )
            + 4 * rho * theta
        )
        - 64 * kappa**6 * rho * sigma * t**2 * theta
        + 32 * kappa**7 * t**2 * theta
        + 2 * kappa * sigma**4 * (
            116 * rho * sigma + 5 * sigma**2 * t + theta
        )
        - 29 * sigma**6
    )
    + sigma**2 * (sigma - 4*kappa*rho)**2 * (2*kappa*theta + sigma**2)
)

# Check equivalence
zeta_2_result = sp.simplify(zeta_2_check - zeta_2) == 0
zeta_3_result = sp.simplify(zeta_3_check - zeta_3) == 0
zeta_4_result = sp.simplify(zeta_4_check - zeta_4) == 0

# Print result
print("Are the zeta_2 equivalent?", zeta_2_result)
print("Are the zeta_3 equivalent?", zeta_3_result)
print("Are the zeta_4 equivalent?", zeta_4_result)