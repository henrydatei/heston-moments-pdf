import numpy as np
import scipy.fft as fft
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

def characteristic_function(u, mu, kappa, theta, sigma, rho, t):
    """
    This function calculates the characteristic function of log-return rt of a Bates process

    Args:
        u: Argument of the characteristic function
        mu: Drift of the price process
        kappa: Rate of mean reversion.
        theta: Long-term mean.
        sigma: Volatility of the variance diffusion process.
        rho: Correlation between Brownian motions.
        t: Time horizon.

    Returns:
        characteristic function of log-return rt
    """

    # Define functions for each component
    A = mu * t * u * 1j

    d = np.sqrt((rho*sigma*u*1j - kappa)**2 - sigma**2*(-u*1j - u**2))

    g = (kappa - rho*sigma*u*1j - d) / (kappa - rho*sigma*u*1j + d)

    B = theta * kappa / sigma**2 * ((kappa - rho*sigma*u*1j - d) * t - 2 * np.log((1-g*np.exp(-d*t)) / (1-g)))

    D = 0j

    gamma = 2 * kappa * theta / sigma**2

    C_unc = np.log((2*kappa/sigma**2)**gamma * (2*kappa/sigma**2 - (kappa - rho*sigma*u*1j - d) / sigma**2 * (1 - np.exp(-d*t)) / (1 - g * np.exp(-d*t)))**(-gamma))


    phi = np.exp(A + B + C_unc + D)

    return phi

def compute_density_via_ifft_simple(mu, kappa, theta, sigma, rho, tau, N=2**15, L=10):
    """
    Computes the density function using the inverse FFT.

    Args:
        mu, kappa, theta, sigma, rho, tau: Parameters of the characteristic function.
        N: Number of FFT points (power of 2).
        L: Limit for the integration range [-L, L].

    Returns:
        x: Grid points.
        f_x: Density function values.
    """

    t = np.linspace(-L, L, N)
    dt = t[1] - t[0]
    
    phi_t = np.array([characteristic_function(tt, mu, kappa, theta, sigma, rho, tau) for tt in t])
    
    phi_t_shifted = phi_t * np.exp(-1j * t * (-L))
    
    f_x = np.fft.ifft(np.fft.fftshift(phi_t_shifted)).real / dt

    return t, f_x

def compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, t):
    i = 1j  # assigning i=sqrt(-1)

    # Define unconditional characteristic function for the log-return rt
    cF = lambda u: characteristic_function(u, mu, kappa, theta, sigma, rho, t)

    # define domain for density
    x = np.linspace(-2.0, 2.0, 1000)

    # recovered density
    f_x = RecoverDensity(cF, x, 2 ** 15)
    
    # Sometimes f_x seems to be slightly negative, so we need to set it to zero
    # f_x = np.maximum(f_x, 0)

    return x, f_x

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


# # plot the Heston log-return distribution
# x1, density_1 = compute_density_via_ifft_simple(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, tau = 1/12)
# x2, density_2 = compute_density_via_ifft_accurate(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, tau = 1/12)

# # print(len(density_1), len(density_2))

# plt.grid()
# plt.xlabel("x")
# plt.ylabel("$f(x)$")
# plt.plot(x1, density_1, color="blue", label='Density with simple FFT')
# plt.plot(x2, density_2, color="green", label='Density with FFT and interpolation and stabilization')
# plt.xlim(-10, 10)
# plt.ylim(0, 5)
# plt.title("Different methods for computing the density")

# # Adding a legend in the upper left corner
# plt.legend()
# plt.tight_layout()
# plt.show()