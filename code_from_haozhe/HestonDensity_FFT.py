# Recover Heston density function by Fast Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
from DensityRecoveryFFT import RecoverDensity, RecoverDensity_MiniError
from CharacteristicFunction import ChF_Bates_ORS, ChF_Bates_Gatheral
from Moments_list import MomentsBates

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
