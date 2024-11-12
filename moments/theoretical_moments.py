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
    mu_3 = 0
    mu_4 = 0
    
    zeta_1 = mu_1
    zeta_2 = theta * (
        -4 * kappa**2 * rho * sigma * t
        + 4 * kappa**3 * t
        + sigma * np.exp(-kappa * t) * (sigma - 4 * kappa * rho)
        + 4 * kappa * sigma * rho
        + kappa * sigma**2 * t
        - sigma**2
    ) / (4 * kappa**3)
    zeta_3 = 0
    zeta_4 = 0
    
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