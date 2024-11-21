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