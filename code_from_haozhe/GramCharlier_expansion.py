import numpy as np

# Hermite Polynomials
def hermite(x, k):
    if k == 0:
        return 1
    elif k == 1:
        return x
    elif k == 2:
        return x**2 - 1
    elif k == 3:
        return x * (x**2 - 3)
    elif k == 4:
        z = x**2
        return z * (z - 6) + 3
    else:
        raise ValueError("Order k > 5")
    
def P4(z, mu, sigma, gamma1, gamma2):         # gamma1 is the skewness while the gamma2 is the excess kurtosis
    return 1 + gamma1 / 6 * hermite((z - mu) / sigma, 3) + gamma2 / 24 * hermite((z - mu) / sigma, 4)

def Phi(z, mu, sigma):             # normal density
    return np.exp(-(z-mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

def TypeA_approximation(P, Phi):
    return P * Phi

def Expansion_GramCharlier(cumulant):

    # classify the inputs
    mu, sigma, k3, k4 = cumulant[0], np.sqrt(cumulant[1]), cumulant[2], cumulant[3]
    gamma1, gamma2 = k3 / sigma**3, k4 / sigma**4 - 3
    z = np.linspace(-2.0, 2.0, 1000)

    # get the expansion
    p = P4(z, mu, sigma, gamma1, gamma2)
    phi = Phi(z, mu, sigma)
    g = TypeA_approximation(p, phi)

    return g
