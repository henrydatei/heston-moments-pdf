from typing import Tuple
import numpy as np

def theoretical_moments_okhrin_20222(mu: float, kappa: float, theta: float, sigma: float, rho: float, t: float) -> Tuple[float, float, float, float, float, float, float, float]:
    mu_1 = (mu - theta / 2) * t
    mu_2 = (1 / (4 * kappa**3)) * (
        np.exp(-kappa * t) * (
            np.exp(kappa * t) * (
                kappa**3 * t * (t * (theta - 2 * mu)**2 + 4 * theta)
                - 4 * kappa**2 * rho * sigma * t * theta
                + kappa * sigma * theta * (4 * rho + sigma * t)
                - sigma**2 * theta
            )
            + sigma * theta * (sigma - 4 * kappa * rho)
        )
    )
    mu_3 = (np.exp(-kappa * t))/(8*kappa**5) * (
        np.exp(kappa*t) * (
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
    mu_4 = (1 / (32 * kappa**7)) * (
        2 * kappa * t * (
            6 * kappa**4 * t**2 * theta * (theta - 2 * mu)**2 * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)
            - 12 * kappa**2 * sigma * t * theta * (theta - 2 * mu) * (2 * kappa * rho - sigma) * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)
            + kappa**6 * t**3 * (theta - 2 * mu)**4
            + 3 * sigma**2 * theta * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2) * (4 * kappa**2 * (4 * rho**2 + 1) - 20 * kappa * rho * sigma + 5 * sigma**2)
            + 3 * kappa**2 * t * theta**2 * (4 * kappa**2 - 4 * kappa * rho * sigma + sigma**2)**2
        )
        - 24 * kappa**2 * sigma * t * theta * np.exp(-kappa * t) * (2 * mu - theta) * (2 * kappa * rho - sigma) * (
            4 * kappa**2 * (np.exp(kappa * t) + rho * sigma * t - 1)
            - kappa * sigma * (8 * rho * (np.exp(kappa * t) - 1) + sigma * t)
            + 2 * sigma**2 * (np.exp(kappa * t) - 1)
        )
        + 12 * kappa**2 * sigma * t * theta * np.exp(-kappa * t) * (np.exp(kappa * t) - 1) * (4 * kappa * rho - sigma) * (
            kappa**2 * (t * (theta - 2 * mu)**2 + 4 * theta)
            - 4 * kappa * rho * sigma * theta
            + sigma**2 * theta
        )
        + 3 * sigma**2 * theta * np.exp(-2 * kappa * t) * (
            -4 * np.exp(kappa * t) * (
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
            ) + np.exp(2 * kappa * t) * (
                32 * kappa**3 * rho * (12 * (rho**2 + 1) * sigma + rho * theta)
                - 16 * kappa**2 * sigma * (35 * rho**2 * sigma + rho * theta + 6 * sigma)
                - 32 * kappa**4 * (8 * rho**2 + 1)
                + 2 * kappa * sigma**2 * (116 * rho * sigma + theta)
                - 29 * sigma**4
            ) + (sigma - 4 * kappa * rho)**2 * (2 * kappa * theta + sigma**2)
        )
    )
    
    zeta_1 = mu_1
    zeta_2 = theta * (
        -4 * kappa**2 * rho * sigma * t
        + 4 * kappa**3 * t
        + sigma * np.exp(-kappa * t) * (sigma - 4 * kappa * rho)
        + 4 * kappa * sigma * rho
        + kappa * sigma**2 * t
        - sigma**2
    ) / (4 * kappa**3)
    zeta_3 = (mu_3 - 3 * mu_1 * mu_2 + 2 * mu_1**3) / (zeta_2**1.5)
    zeta_4 = (mu_4 - 4 * mu_1 * mu_3 + 6 * mu_1**2 * mu_2 - 3 * mu_1**4) / (zeta_2**2)
    
    return mu_1, mu_2, mu_3, mu_4, zeta_1, zeta_2, zeta_3, zeta_4

def theoretical_moments_dunn_2014(mu: float, kappa: float, theta: float, sigma: float, rho: float) -> Tuple[float, float, float, float, float, float, float, float]:
    mu_1 = 1 + mu
    mu_2 = (mu + 1)**2 + theta
    mu_3 = (mu + 1)**3 + 3*theta + 3*mu*theta
    mu_4 = 1/(kappa * (kappa - 2)) * (kappa**2 * mu**4 + 4 * kappa**2 * mu**3 + 6 * kappa**2 * mu**2 * theta - 2 * kappa * mu**4 + 6 * kappa**2 * mu**2 + 12 * kappa**2 * mu * theta + 3 * kappa**2 * theta**2 - 8 * kappa * mu**3 - 12 * kappa * mu**2 * theta + 4 * kappa**2 * mu + 6 * kappa**2 * theta - 12 * kappa * mu**2 - 24 * kappa * mu * theta - 6 * kappa * theta**2 - 3 * sigma**2 * theta + kappa**2 - 8 * kappa * mu - 12 * kappa * theta - 2 * kappa)
    
    zeta_1 = mu_1
    zeta_2 = theta
    zeta_3 = 0
    zeta_4 = 3 * (kappa**2 * theta - 2 * kappa * theta - sigma**2)/(kappa * theta * (kappa - 2))
    
    return mu_1, mu_2, mu_3, mu_4, zeta_1, zeta_2, zeta_3, zeta_4