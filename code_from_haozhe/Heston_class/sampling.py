import numpy as np
from scipy.stats import norm
from pytorch_lightning import seed_everything

class Sampling:
    
    def __init__(self, heston_instance):
        self.heston_instance = heston_instance  # Reference to the Heston instance        

    # simulation of the Heston model using QE scheme
    # v - CIR process, can be given in order to get only prices with the given spot vola
    # Z_v - used for other simulation methods in order to have a noise for the CIR process
    def rQE(self, n, TT, M = 1, v0 = None, v = None, Z_v = None, gam1 = 0.5, gam2 = 0.5):
        #seed_everything(seed=33)
        mu = self.heston_instance.mu
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        #T = self.heston_instance.T
        TT = TT       # total time: simulation horizon in years
        S0 = self.heston_instance.S0
        
        if v0 is None:
            v0 = theta

        # Define compulsory placeholders
        lnSt = np.zeros((M, n + 1))
        lnSt[:, 0] = np.log(S0)
        
        # Define some constants
        Delta = TT / n
        
        U = np.round(np.random.uniform(low=1e-10, high=1 - 1e-10, size=(M, n)), 10)  #  generate uniform random numbers strictly in the interval (0,1)
        
        psic = 1.5
        
        emkt = np.exp(-kappa * Delta)
        c1  = sigma * sigma * emkt * (1 - emkt) / kappa
        c2  = theta * sigma * sigma * ((1 - emkt)**2) / (2 * kappa)
        
        #gam1 = 0.5
        #gam2 = 0.5

        K0 = -rho * kappa * theta * Delta / sigma
        K1a = Delta * (kappa * rho / sigma - 0.5)
        K1b = rho / sigma
        K1 = gam1 * K1a - K1b
        K2 = gam2 * K1a + K1b
        Kc = Delta * (1 - rho * rho)
        K3 = gam1 * Kc
        K4 = gam2 * Kc

        #if Z_v is None:
        #    Z_v = np.random.randn(M, n)
        # If necessary, simulate CIR process. Do so by QE.
        if v is None:
            v   = np.zeros((M, n + 1))
            v[:, 0]= v0

            for i in range(n):
        
                s2  = v[:, i] * c1 + c2
                m   = v[:, i] * emkt + theta * (1 - emkt)
                m2 = m ** 2
                psi = s2 / m2
                #psi = s2 / (m * m)
                two_psi_i = 2 * m2 / s2
                two_psi_pl1_inv = 2 * m2 / (s2 + m2)
                bigger_term = two_psi_i + np.sqrt(two_psi_i**2 - two_psi_i)

                # need to be checked if I did not make any mistake while simplifying
                v[:, i + 1] = np.where(
                    psi <= psic,
                    m / bigger_term * (np.sqrt(bigger_term - 1) + norm.ppf(U[:, i]))**2,
                    #m / (1 + 2 * 1/psi - 1 + np.sqrt(2 * 1/psi) * np.sqrt(2 * 1/psi - 1)) * (np.sqrt(2 * 1/psi - 1 + np.sqrt(2 * 1/psi) * np.sqrt(2 * 1/psi - 1)) + norm.ppf(U[:, i]))**2,                    
                    np.where(U[:, i] < 1 - two_psi_pl1_inv, 0, (two_psi_pl1_inv / m)**(-1) * np.log(two_psi_pl1_inv / (1 - U[:, i])))
                    #np.where(U[:, i] < (psi - 1) / (psi + 1), 0, ((1 - (psi - 1) / (psi + 1)) / m)**(-1) * np.log((1 - (psi - 1) / (psi + 1)) / (1 - U[:, i])))
                )
                
        # do the simulations for log-price
        Z_x = np.random.randn(M, n)
        for i in range(n):
            lnSt[:, i + 1] = lnSt[:, i] + mu * Delta + K0 + K1 * v[:, i] + K2 * v[:, i+1] + np.sqrt(K3 * v[:, i] + K4 * v[:, i+1]) * Z_x[:, i]

        # Filter out invalid rows
        valid_rows = np.all(np.isfinite(lnSt), axis=1)      # check whether each row contains any nan or inf element
        filtered_lnSt = lnSt[valid_rows]

        # re-draw new paths if some are invalid and log number of rows removed if not verbose
        rows_removed = lnSt.shape[0] - filtered_lnSt.shape[0]
        if rows_removed > 0:
            # Re-generate missing rows
            new_paths = self.rQE(n, TT, rows_removed, v0=v0, v=None, Z_v=Z_v, gam1=gam1, gam2=gam2)

            # Append new paths to filtered_lnSt
            filtered_lnSt = np.vstack([filtered_lnSt, new_paths])

            # Debug/logging
            if not self.heston_instance.verbose:
                print(f"{rows_removed} simulation paths were invalid and replaced with new paths.")

        # Ensure the output shape is (M, n + 1)
        assert filtered_lnSt.shape == (M, n + 1), "Output shape mismatch after filtering and replacement."

        return filtered_lnSt  