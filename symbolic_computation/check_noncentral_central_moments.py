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
Ert2 = (sigma * (- 4 * kappa * rho + sigma) * theta + sp.exp(kappa * t) * (- (sigma**2 * theta) - 4 * kappa**2 * rho * sigma * t * theta + kappa * sigma * (4 * rho + sigma * t) * theta + kappa**3 * t * (4 * theta + t * (-2 * mu + theta)**2))) / (4 * sp.exp(kappa * t) * kappa**3)

mu_3 = (sp.exp(-kappa * t))/(8*kappa**5) * (
    sp.exp(kappa*t) * (
        3 * kappa**2 * sigma**2 * theta * (
            -2*mu*t + 16*rho**2 + 6*rho*sigma*t + t*theta + 4
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
Ert3 =  (- 3 * sigma * theta * (2 * sigma**3 + kappa * sigma**2 * (- 12 * rho + sigma * t) + 4 * kappa**3 * rho*(- 2 + 2 * mu * t + 2 * rho * sigma * t - t * theta) + 
kappa**2 * sigma * (4 + 16 * rho**2 - 2 * mu * t - 6 * rho * sigma * t + t * theta)) + 
sp.exp(kappa * t) * (6 * sigma**4 * theta - 3 * kappa * sigma**3 * (12 * rho + sigma * t) * theta + 12 * kappa**4 * rho * sigma * t * theta * (2 - 2 * mu * t + t * theta) + 
3 * kappa**2 * sigma**2 * theta * (4 + 16 * rho**2 - 2 * mu * t + 6 * rho * sigma * t + t * theta) - 
3 * kappa**3 * sigma * theta * (8 * rho**2 * sigma * t + sigma * t * (4 - 2 * mu * t + t * theta) + rho * (8 - 8 * mu * t + 4 * t * theta)) + 
kappa**5 * t * (t * (2 * mu - theta) * (12 * theta + t * (- 2 * mu + theta)**2)))) / (8 * sp.exp(kappa * t) * kappa**5)

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
Ert4 = (( - 24 * kappa**2 * (2 * kappa * rho - sigma) * sigma * t * (2 * (- 1 + sp.exp(kappa * t)) * sigma**2 - kappa * sigma * (8 * (- 1 + sp.exp(kappa * t)) * rho + sigma * t) 
+ 4 * kappa**2 * (- 1 + sp.exp(kappa * t) + rho * sigma * t)) * (2 * mu - theta) * theta)/sp.exp(kappa * t) + (3 * sigma**2 * theta * ((-4 * kappa*rho + sigma)**2 * 
(sigma**2 + 2 * kappa * theta) + sp.exp(2 * kappa * t) * (- 32 * kappa**4 * (1 + 8 * rho**2) - 29 * sigma**4 + 2 * kappa * sigma**2 * (116 * rho * sigma + theta) - 
16 * kappa**2 * sigma * (6 * sigma + 35 * rho**2 * sigma + rho * theta) + 32 * kappa**3 * rho * (12 * (1 + rho**2) * sigma + rho * theta)) - 
4 * sp.exp(kappa * t) * (- 7 * sigma**4 + 16 * kappa**5 * rho**2 * t * (-2 + rho * sigma * t) + 4 * kappa**4 * (- 2 + rho * (12 * sigma*t +
rho * (- 16 + sigma * t * (16 * rho - 5 * sigma * t)))) + kappa * sigma**2 * (56 * rho * sigma - 5 * sigma**2 * t + theta) - 
kappa**2 * sigma * (sigma * (24 + 136 * rho**2 - 40 * rho * sigma * t + sigma**2 * t**2) + 8 * rho * theta) + 
4 * kappa**3 * (sigma * (24 * (rho + rho**3) - 3 * (1 + 8 * rho**2) * sigma * t + 2 * rho * sigma**2 * t**2) + 4 * rho**2 * theta)))) / sp.exp(2 * kappa * t) + (12 * (- 1 + sp.exp(kappa * t)) * kappa**2 * (4 * kappa * rho - sigma) * sigma * t * theta * (- 4 * kappa * rho * sigma * theta + 
sigma**2 * theta + kappa**2 * (4 * theta + t * (- 2 * mu + theta)**2))) / sp.exp(kappa * t) + 
2 * kappa * t * (- 120 * kappa * rho * sigma**5 * theta + 15 * sigma**6 * theta - 48 * kappa**3 * rho * sigma**3 * theta * (6 + 4 * rho**2 - 3 * mu * t + 2 * t * theta) + 
3 * kappa**2 * sigma**4 * theta * (24 + 96 * rho**2 - 8 * mu * t + 5 * t * theta) + kappa**6 * (t * (16 * mu**4 * t**2 - 32 * mu**2 * t * (- 3 + mu * t) * theta +
24 * (2 + mu * t * (- 4 + mu * t)) * theta**2 - 8 * t * (- 3 + mu * t) * theta**3 + t**2 * theta**4)) - 24 * kappa**5 * rho * sigma * t * theta * (4 * mu**2 * t - 4 * mu * (2 + t * theta) + theta * (8 + t * theta)) + 
6 * kappa**4 * sigma**2 * theta * (8 - 16 * mu * t + 8 * rho**2 * (4 - 4 * mu * t + 3 * t * theta) + t * (12 * theta + t * (- 2 * mu + theta)**2)))) / (32 * kappa**7)

# Calculate the zetas
zeta_2_check = mu_2 - mu_1**2
zeta_3_check = (mu_3 - 3 * mu_1 * mu_2 + 2 * mu_1**3) / (zeta_2_check**(3/2))
zeta_4_check = (mu_4 - 4 * mu_1 * mu_3 + 6 * mu_1**2 * mu_2 - 3 * mu_1**4) / (zeta_2_check**2)

# Calculate the zetas with Erts
zeta_2_check_Ert = Ert2 - mu_1**2
zeta_3_check_Ert = (Ert3 - 3 * mu_1 * Ert2 + 2 * mu_1**3) / (zeta_2_check_Ert**(3/2))
zeta_4_check_Ert = (Ert4 - 4 * mu_1 * Ert3 + 6 * mu_1**2 * Ert2 - 3 * mu_1**4) / (zeta_2_check_Ert**2)

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

# Print result
print("Are zeta_2 and zeta_2_check equivalent?", sp.simplify(zeta_2_check - zeta_2) == 0)
print("Are zeta_3 and zeta_3_check equivalent?", sp.simplify(zeta_3_check - zeta_3) == 0)
print("Are zeta_4 and zeta_4_check equivalent?", sp.simplify(zeta_4_check - zeta_4) == 0)

print("Are mu_2 and Ert2 equivalent?", sp.simplify(mu_2 - Ert2) == 0)
print("Are mu_3 and Ert3 equivalent?", sp.simplify(mu_3 - Ert3) == 0)
print("Are mu_4 and Ert4 equivalent?", sp.simplify(mu_4 - Ert4) == 0)

# print("Are zeta_2 and zeta_2_check_Ert equivalent?", sp.simplify(zeta_2 - zeta_2_check_Ert) == 0)
# print("Are zeta_3 and zeta_3_check_Ert equivalent?", sp.simplify(zeta_3 - zeta_3_check_Ert) == 0)
# print("Are zeta_4 and zeta_4_check_Ert equivalent?", sp.simplify(zeta_4 - zeta_4_check_Ert) == 0)

# print("Are zeta_2_check and zeta_2_check_Ert equivalent?", sp.simplify(zeta_2_check - zeta_2_check_Ert) == 0)
# print("Are zeta_3_check and zeta_3_check_Ert equivalent?", sp.simplify(zeta_3_check - zeta_3_check_Ert) == 0)
# print("Are zeta_4_check and zeta_4_check_Ert equivalent?", sp.simplify(zeta_4_check - zeta_4_check_Ert) == 0)

# Difference between zetas and zetas_check
# print(sp.simplify(zeta_3 - zeta_3_check)) # really long expression
# print(sp.simplify(zeta_4 - zeta_4_check)) # an even longer expression

# Numerical evaluation
values = {mu: 0, theta: 0.19, kappa: 3, rho: -0.7, sigma: 0.4, t: 1}
print(zeta_2.subs(values).evalf(), zeta_2_check.subs(values).evalf())
print(zeta_3.subs(values).evalf(), zeta_3_check.subs(values).evalf())
print(zeta_4.subs(values).evalf(), zeta_4_check.subs(values).evalf())