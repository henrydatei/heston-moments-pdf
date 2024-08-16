import numpy as np
from scipy.stats import norm

def Heston_QE(S0, v0, kappa, theta, sigma, mu, rho, T, n, M, v=None, Z_v=None):

    # Define compulsory placeholders
    lnSt = np.zeros((M, n + 1))
    lnSt[:, 0] = np.log(S0)
    
    # Define some constants
    Delta = T / n
    
    U = np.round(np.random.uniform(low=0, high=1, size=(M,n)), 10)  # use round to 10 digits to include the upper bound 1
    
    psic = 1.5
    
    emkt = np.exp(- kappa * Delta)
    c1  = sigma * sigma * emkt * (1 - emkt) / kappa
    c2  = theta * sigma * sigma * ((1 - emkt)**2) / (2 * kappa)
    
    gam1 = 0.5
    gam2 = 0.5

    K0 = -rho * kappa * theta * Delta / sigma
    K1a = Delta * (kappa * rho / sigma - 0.5)
    K1b = rho / sigma
    K1 = gam1 * K1a - K1b
    K2 = gam2 * K1a + K1b
    Kc = Delta * (1 - rho * rho)
    K3 = gam1 * Kc
    K4 = gam2 * Kc

    if Z_v is None:
        Z_v = np.random.randn(M, n)
    # If necessary, simulate CIR process. Do so by QE.
    if v is None:
        v   = np.zeros((M, n + 1))
        v[:, 0]= v0

        for i in range(n):
      
            s2  = v[:, i] * c1 + c2
            m   = v[:, i] * emkt + theta * (1 - emkt)
            psi = s2 / (m * m)
        
            v[:, i + 1] = np.where(
                psi <= psic,
                m / (1 + 2 * 1/psi - 1 + np.sqrt(2 * 1/psi) * np.sqrt(2 * 1/psi - 1)) * (np.sqrt(2 * 1/psi - 1 + np.sqrt(2 * 1/psi) * np.sqrt(2 * 1/psi - 1)) + norm.ppf(U[:, i]))**2,
                np.where(U[:, i] < (psi - 1) / (psi + 1), 0, ((1 - (psi - 1) / (psi + 1)) / m)**-1 * np.log((1 - (psi - 1) / (psi + 1)) / (1 - U[:, i])))
            )

    # do the simulations for log-price
    Z_x = np.random.randn(M, n)
    for i in range(n):
        lnSt[:, i + 1] = lnSt[:, i] + mu * Delta + K0 + K1 * v[:, i] + K2 * v[:, i+1] + np.sqrt(K3 * v[:, i] + K4 * v[:, i+1]) * Z_x[:, i]

    return lnSt