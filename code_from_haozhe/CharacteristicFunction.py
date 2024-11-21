import numpy as np
import math

def ChF_Bates_ORS(u, mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False):
    """
    This function calculates the characteristic function of log-return rt of a Bates process

    Args:
        u: Argument of the characteristic function
        mu: Drift of the price process
        kappa: Rate of mean reversion.
        theta: Long-term mean.
        sigma: Volatility of the variance diffusion process.
        rho: Correlation between Brownian motions.
        lambdaj: Intensity parameter for jumps.
        muj: Mean of the jump size J which is log-normally distributed.
        vj: Volatility of the jump size J which is log-normally distributed.
        t: Time horizon.
        v0: Initial volatility.

    Returns:
        characteristic function of log-return rt
    """

    # Input validation
    if not all([kappa > 0, theta > 0, sigma > 0, -1 <= rho <= 1, lambdaj >= 0, vj >= 0, t >= 0]):
        raise ValueError("Invalid parameters")

    # Define functions for each component
    A = mu * t * u * 1j

    d = np.sqrt((rho*sigma*u*1j - kappa)**2 + sigma**2*(u*1j + u**2))

    g = (kappa - rho*sigma*u*1j - d) / (kappa - rho*sigma*u*1j + d)

    B = theta * kappa / sigma**2 * ((kappa - rho*sigma*u*1j - d) * t - 2 * np.log((1-g*np.exp(-d*t)) / (1-g)))

    D = -lambdaj * (np.log(1+muj) - vj**2) * u * t * 1j + lambdaj * t * ((1+muj)**(u*1j) * np.exp(vj**2 * u*1j * (u*1j-1)) - 1)

    gamma = 2 * kappa * theta / sigma**2

    C_v0 = v0 * (kappa - rho*sigma*u*1j - d) / sigma**2 * (1 - np.exp(-d*t)) / (1 - g * np.exp(d*t))

    C_unc = np.log((2*kappa/sigma**2)**gamma * (2*kappa/sigma**2 - (kappa - rho*sigma*u*1j - d) / sigma**2 * (1 - np.exp(-d*t)) / (1 - g * np.exp(d*t)))**(-gamma))

    if conditional:

        phi = np.exp(A + B + C_v0 + D)

    else:

        phi = np.exp(A + B + C_unc + D)

    return phi

def ChF_Bates_Gatheral(u, mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False):
    """
    This function calculates the characteristic function of log-return rt of a Bates process

    Args:
        u: Argument of the characteristic function
        mu: Drift of the price process
        kappa: Rate of mean reversion.
        theta: Long-term mean.
        sigma: Volatility of the variance diffusion process.
        rho: Correlation between Brownian motions.
        lambdaj: Intensity parameter for jumps.
        muj: Mean of the jump size J which is log-normally distributed.
        vj: Volatility of the jump size J which is log-normally distributed.
        t: Time horizon.
        v0: Initial volatility.

    Returns:
        characteristic function of log-return rt
    """

    # Input validation
    if not all([kappa > 0, theta > 0, sigma > 0, -1 <= rho <= 1, lambdaj >= 0, vj >= 0, t >= 0]):
        raise ValueError("Invalid parameters")

    # Define functions for each component
    A = mu * t * u * 1j

    d = np.sqrt((rho*sigma*u*1j - kappa)**2 - sigma**2*(-u*1j - u**2))

    g = (kappa - rho*sigma*u*1j - d) / (kappa - rho*sigma*u*1j + d)

    B = theta * kappa / sigma**2 * ((kappa - rho*sigma*u*1j - d) * t - 2 * np.log((1-g*np.exp(-d*t)) / (1-g)))

    D = -lambdaj * (np.log(1+muj) - vj**2) * u * t * 1j + lambdaj * t * ((1+muj)**(u*1j) * np.exp(vj**2 * u*1j * (u*1j-1)) - 1)

    gamma = 2 * kappa * theta / sigma**2

    C_v0 = v0 * (kappa - rho*sigma*u*1j - d) / sigma**2 * (1 - np.exp(-d*t)) / (1 - g * np.exp(-d*t))

    C_unc = np.log((2*kappa/sigma**2)**gamma * (2*kappa/sigma**2 - (kappa - rho*sigma*u*1j - d) / sigma**2 * (1 - np.exp(-d*t)) / (1 - g * np.exp(-d*t)))**(-gamma))
   

    if conditional:

        phi = np.exp(A + B + C_v0 + D)

    else:

        phi = np.exp(A + B + C_unc + D)

    return phi