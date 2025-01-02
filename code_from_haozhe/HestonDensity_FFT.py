# Recover Heston density function by Fast Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
# from DensityRecoveryFFT import RecoverDensity, RecoverDensity_MiniError
# from CharacteristicFunction import ChF_Bates_ORS, ChF_Bates_Gatheral
# from Moments_list import MomentsBates

# Recover density function by Fast Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.fft as fft
import scipy.interpolate as interpolate


def RecoverDensity(cf, x, N=8192):
    i = 1j  # assigning i=sqrt(-1)

    # specification of the grid for u
    u_max = 1000.0
    du = u_max / N
    u = np.linspace(0, N - 1, N) * du

    # grid for x
    b = np.min(x)
    dx = 2.0 * np.pi / (N * du)
    x_i = b + np.linspace(0, N - 1, N) * dx

    phi = np.exp(-i * b * u) * cf(u)         # here takes negative sign because b is with positive sign

    gamma_1 = np.exp(-i * x_i * u[0]) * cf(u[0])
    gamma_2 = np.exp(-i * x_i * u[-1]) * cf(u[-1])

    phi_boundary = 0.5 * (gamma_1 + gamma_2)

    f_xi = du / np.pi * np.real(fft.fft(phi) - phi_boundary)

    f_xiInterp = interpolate.interp1d(x_i, f_xi, kind='cubic')

    return f_xiInterp(x)

def RecoverDensity_MiniError(cf, x, u_epsilon):
    i = 1j  # assigning i=sqrt(-1)

    # specification of the grid for u
    u_max = 1000.0

    # specify N 
    N = 1000       # initial guess
    epsilon = 1e-5
    h = 2 * np.pi / (6 + u_epsilon)        # h will be determined by mean + 5*std
    while np.abs(cf(h * N) / N) > np.pi * epsilon / 2:
        N += 1

    du = h
    u = np.linspace(0, N - 1, N) * du

    # grid for x
    b = np.min(x)
    dx = 2.0 * np.pi / (N * du)
    x_i = b + np.linspace(0, N - 1, N) * dx

    phi = np.exp(-i * b * u) * cf(u)         # here takes negative sign because b is with positive sign

    gamma_1 = np.exp(-i * x_i * u[0]) * cf(u[0])
    gamma_2 = np.exp(-i * x_i * u[-1]) * cf(u[-1])

    phi_boundary = 0.5 * (gamma_1 + gamma_2)

    f_xi = du / np.pi * np.real(fft.fft(phi) - phi_boundary)

    f_xiInterp = interpolate.interp1d(x_i, f_xi, kind='cubic')

    return f_xiInterp(x)

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


def HestonChfDensity_FFT(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False):
    i = 1j  # assigning i=sqrt(-1)

    # Define unconditional characteristic function for the log-return rt
    cF = lambda u: ChF_Bates_ORS(u, mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = conditional)

    # define domain for density
    x = np.linspace(-2.0, 2.0, 1000)

    # recovered density
    f_x = RecoverDensity(cF, x, 2 ** 15)

    return f_x

def HestonChfDensity_FFT_Gatheral(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False):
    i = 1j  # assigning i=sqrt(-1)

    # Define unconditional characteristic function for the log-return rt
    cF = lambda u: ChF_Bates_Gatheral(u, mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = conditional)

    # define domain for density
    x = np.linspace(-2.0, 2.0, 1000)

    # recovered density
    f_x = RecoverDensity(cF, x, 2 ** 15)

    return f_x

# def HestonChfDensity_FFT_MiniError(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False):
#     i = 1j  # assigning i=sqrt(-1)

#     # Define unconditional characteristic function for the log-return rt
#     cF = lambda u: ChF_Bates(u, mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = False)

#     # Get true cumulants
#     cumulants = MomentsBates(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional=False, nc=False)
#     mean, var = cumulants[0], cumulants[1]
#     u_epsilon = np.abs(mean + 5 * np.sqrt(var))

#     # define domain for density
#     x = np.linspace(-1.0, 1.0, 1000)

#     # recovered density
#     f_x = RecoverDensity_MiniError(cF, x, u_epsilon)

#     return f_x


# # plot the Heston log-return distribution
# f_x_ors_unc = HestonChfDensity_FFT(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)
# f_x_ors_c = HestonChfDensity_FFT(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=True)
# f_x_g_unc = HestonChfDensity_FFT_Gatheral(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)
# f_x_g_c = HestonChfDensity_FFT_Gatheral(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=True)
# #f_x_2 = HestonChfDensity_FFT_MiniError(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)
# x = np.linspace(-2.0, 2.0, 1000)
# plt.figure(figsize=(12, 8))
# plt.grid()
# plt.xlabel("x")
# plt.ylabel("$f(x)$")
# plt.plot(x, f_x_ors_unc, '--', color="blue", label='Unconditional Density from ORS')
# plt.plot(x, f_x_ors_c, '.', color="cyan", label='Conditional Density from ORS')
# plt.plot(x, f_x_g_unc, '--', color="green", label='Unconditional Density from Gatheral')
# plt.plot(x, f_x_g_c, '.', color="olive", label='Conditional Density from Gatheral')

# # Adding a legend in the upper left corner
# plt.legend(loc='upper left')
# plt.savefig('Heston_Density.pdf') 
# plt.show()
