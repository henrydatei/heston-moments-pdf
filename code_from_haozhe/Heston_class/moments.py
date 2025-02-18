import numpy as np

"""
Moments Class
=============

The `Moments` class computes the moments of financial models based on the Heston and Bates frameworks. These metrics provide insights into the asymmetry (skewness) and peakedness (kurtosis) of the distribution of asset returns over a specified time horizon. The class interacts with an instance of the Heston model and uses its parameters to calculate the moments.

Class Methods
-------------
1. `Meanrt()`: Computes the theoretical mean of the returns.
2. `Varrt()`: Computes the theoretical variance of the returns.
3. `Skewrt()`: Computes the theoretical skewness of the returns.
4. `Kurtrt()`: Computes the theoretical kurtosis of the returns.

Private Methods
---------------
1. `__Ert2_Heston()`: Computes the second noncentralized moment for the pure Heston model.
2. `__Ert2_Bates()`: Computes the second noncentralized moment for the Bates model, which extends the Heston model with jump components.
3. `__Skewrt_Heston()`: Computes skewness for the pure Heston model.
4. `__Skewrt_Bates()`: Computes skewness for the Bates model, which extends the Heston model with jump components.
5. `__Kurtrt_Heston()`: Computes kurtosis for the pure Heston model.
6. `__Kurtrt_Bates()`: Computes kurtosis for the Bates model.

Attributes
----------
`heston_instance` : object
    An instance of a Heston model, providing parameters such as `theta`, `kappa`, `sigma`, `rho`, `vj`, `muj`, `lambda_`, and `T`.

Usage
-----
To use the `Moments` class, you need to instantiate it with a Heston model object:

    from moments import Moments
    heston_instance = HestonModel(theta, kappa, sigma, rho, vj, muj, lambda_, T)
    moments = Moments(heston_instance)

    # Compute skewness
    skewness = moments.Skewrt()

    # Compute kurtosis
    kurtosis = moments.Kurtrt()

Method Details
--------------
1. `Skewrt()`:
    - Determines whether the Bates model is required based on `lambda_`, `muj`, and `vj`.
    - Calls `__Skewrt_Heston()` if no jump components are present.
    - Calls `__Skewrt_Bates()` if jump components are included.

2. `Kurtrt()`:
    - Similar to `Skewrt()`, decides whether to use the Heston or Bates model for kurtosis calculation.
    - Calls `__Kurtrt_Heston()` if no jump components are present.
    - Calls `__Kurtrt_Bates()` if jump components are included.

Dependencies
------------
This class relies on the following:
- `numpy` for mathematical operations.
- A valid instance of the Heston model.

Parameters from `heston_instance`
---------------------------------
- `theta` : Long-run variance.
- `kappa` : Mean reversion speed.
- `sigma` : Volatility of volatility.
- `rho` : Correlation between asset price and variance.
- `vj` : Jump variance.
- `muj` : Mean of jumps.
- `lambda_` : Jump intensity.
- `T` : Time interval to consider a return.

Notes
-----
- Ensure the Heston model instance is correctly initialized with all necessary parameters before using the `Moments` class.
- This implementation assumes the Bates model's jump components (if present) follow specific conditions defined by the input parameters.
"""

class Moments:
    
    def __init__(self, heston_instance):
        self.heston_instance = heston_instance  # Reference to the Heston instance        

    # computation of the moments of the Heston model
    def __Meanrt_Bates(self):
        mu = self.heston_instance.mu
        theta = self.heston_instance.theta
        T = self.heston_instance.T

        result = (mu - theta/2)*T
        return result

    def __Ert2_Bates(self):
        mu = self.heston_instance.mu
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        T = self.heston_instance.T

        sigma2 = sigma ** 2
        kappa2 = kappa ** 2
        kappa3 = kappa ** 3

        numerator = (sigma * (- 4 * kappa * rho + sigma) * theta + np.exp(kappa * T) * (- (sigma2 * theta) - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * (4 * rho + sigma * T) * theta + 
                                                                                        kappa3 * T * (4 * theta + T * (-2 * mu + theta)**2 + lambda_ * vj * (4 + vj)) + 4 * kappa3 * lambda_ * T * np.log(1 + muj) * (-vj + np.log(1 + muj))))

        denominator = (4 * np.exp(kappa * T) * kappa3)

        result = numerator / denominator
        return result

    def __Ert2_Heston(self):
        mu = self.heston_instance.mu
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        T = self.heston_instance.T

        sigma2 = sigma ** 2
        kappa2 = kappa ** 2
        kappa3 = kappa ** 3

        numerator = (sigma * (- 4 * kappa * rho + sigma) * theta + np.exp(kappa * T) * (- (sigma2 * theta) - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * (4 * rho + sigma * T) * theta + 
                                                                                        kappa3 * T * (4 * theta + T * (-2 * mu + theta)**2)))

        denominator = (4 * np.exp(kappa * T) * kappa3)

        result = numerator / denominator
        return result
    
    def Meanrt(self):
        return self.__Meanrt_Bates()

    def Varrt(self):
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        
        if (lambda_ == 0) & (muj == 0) & (vj == 0):
            return self.__Ert2_Heston() - (self.Meanrt()**2)
        else:
            return self.__Ert2_Bates() - (self.Meanrt()**2)

    def __Skewrt_Bates(self):
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        T = self.heston_instance.T
        #np = self.heston_instance.np # numpy package
        
        sigma2 = sigma ** 2
        kappa2 = kappa ** 2
        kappa3 = kappa ** 3
        kappa5 = kappa ** 5

        numerator = (-3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + kappa * sigma * (8 * rho - sigma * T) + 4 * kappa2 * (-1 + rho * sigma * T)) * theta - np.exp(kappa * T) * (-3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + 4 * kappa3 * T + kappa * sigma * (8 * rho + sigma * T) - 4 * kappa2 * (1 + rho * sigma * T)) * theta + 12 * kappa5 * lambda_ * T * vj**2 + kappa5 * lambda_ * T * vj**3) + 2 * np.exp(kappa * T) * kappa5 * lambda_ * T * np.log(1 + muj) * (3 * vj * (4 + vj) - 6 * vj * np.log(1 + muj) + 4 * (np.log(1 + muj)**2)))

        denominator = (np.exp(kappa * T) * kappa5 * (((-1 + np.exp(-kappa * T)) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * ((4 - 4 / np.exp(kappa * T)) * rho + sigma * T) * theta + kappa3 * T * (4 * theta + lambda_ * vj * (4 + vj))) / kappa3 - 4 * lambda_ * T * vj * np.log(1 + muj) + 4 * lambda_ * T * (np.log(1 + muj)**2)) ** 1.5)

        result = numerator / denominator
        return result

    def __Skewrt_Heston(self):
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        T = self.heston_instance.T            
        #np = self.heston_instance.np # numpy package
        
        sigma2 = sigma ** 2
        kappa2 = kappa ** 2
        kappa3 = kappa ** 3
        kappa5 = kappa ** 5

        numerator = -3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + kappa * sigma * (8 * rho - sigma * T) + 4 * kappa2 * (-1 + rho * sigma * T)) * theta - np.exp(kappa * T) * (-3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + 4 * kappa3 * T + kappa * sigma * (8 * rho + sigma * T) - 4 * kappa2 * (1 + rho * sigma * T)) * theta)

        denominator = (np.exp(kappa * T) * kappa5 * (((-1 + np.exp(-kappa * T)) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * ((4 - 4 / np.exp(kappa * T)) * rho + sigma * T) * theta + kappa3 * T * 4 * theta) / kappa3) ** 1.5)

        result = numerator / denominator
        return result

    def Skewrt(self):
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        
        if (lambda_ == 0) & (muj == 0) & (vj == 0):
            return self.__Skewrt_Heston()
        else:
            return self.__Skewrt_Bates()
        

    def __Kurtrt_Heston(self):
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        T = self.heston_instance.T
        #np = self.heston_instance.np # numpy package
        
        sigma2 = sigma**2  
        kappa2 = kappa**2 
        kappa3 = kappa**3
        kappa4 = kappa**4
        kappa7 = kappa**7
        exp_kappa_t = np.exp(kappa * T)
        #lambdaj_vj = lambda_ * vj
        theta_lambdaj_vj = 4 * theta

        Kurtrt = (3 * sigma2 * (- 4 * kappa * rho + sigma)**2 * theta * (sigma2 + 2 * kappa * theta) + 12 * exp_kappa_t * sigma * theta * (7 * sigma**5 - 
            kappa * sigma**3 * (56 * rho*sigma - 5 * sigma2 * T + theta) + kappa2 * sigma2 * (- 40 * rho * sigma2 * T + sigma**3 * T**2 + 
            8 * rho * theta + sigma * (24 + 136 * rho**2 + T * theta)) - 4 * kappa3 * sigma * (24 * rho**3 * sigma - 3 * sigma2 * T + 4 * rho**2 * (- 6 * sigma2 * T + theta) + 
            2 * rho * sigma * (12 + sigma2 * T**2 + T * theta)) - 4 * kappa**5 * rho * T * (- 8 * rho * sigma + 4 * rho**2 * sigma2 * T + theta_lambdaj_vj) + 
            kappa4 * sigma * (8 - 48 * rho * sigma * T - 64 * rho**3 * sigma * T + 4 * rho**2 * (16 + 5 * sigma2 * T**2 + 4 * T * theta) + T * theta_lambdaj_vj)) + 
            exp_kappa_t**2 * (- 87 * sigma**6 * theta + 6 * kappa * sigma**4 * theta * (116 * rho * sigma + 5 * sigma2 * T + theta) + 6 * kappa3 * sigma2 * theta * (192 * rho**3 * sigma + 
            16 * rho**2 * (6 * sigma2 * T + theta) + 16 * rho * sigma * (12 + T * theta) + sigma2 * T * (24 + T*theta)) - 12 * kappa2 * sigma**3 * theta * (20 * rho * sigma2 * T + 
            4 * rho * theta + sigma * (24 + 140 * rho**2 + T * theta)) - 48 * kappa**6 * rho * sigma * T**2 * theta * theta_lambdaj_vj - 
            12 * kappa4 * sigma2 * theta * (8 + 32 * rho**3 * sigma * T + 16 * rho**2 * (4 + T * theta) + 4 * rho * sigma * T * (12 + T * theta) + 
            T * theta_lambdaj_vj) + 2 * kappa7 * T * (3 * T * theta_lambdaj_vj**2) + 
            12 * kappa**5 * sigma * T * theta * (4 * rho * theta_lambdaj_vj + sigma * (8 + 8 * rho**2 * (4 + T * theta) + T * theta_lambdaj_vj)))) / (2 * exp_kappa_t**2 * kappa7 * (((- 1 + 1 / exp_kappa_t) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + 
            kappa * sigma * ((4 - 4 / exp_kappa_t) * rho + sigma * T) * theta + kappa3 * T * theta_lambdaj_vj) / kappa3)**2)
    
        return Kurtrt
        
    def __Kurtrt_Bates(self):
        theta = self.heston_instance.theta
        kappa = self.heston_instance.kappa
        sigma = self.heston_instance.sigma
        rho = self.heston_instance.rho
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        T = self.heston_instance.T            
        #np = self.heston_instance.np # numpy package
        
        sigma2 = sigma**2  
        kappa2 = kappa**2 
        kappa3 = kappa**3
        kappa4 = kappa**4
        kappa7 = kappa**7
        log_muj = np.log(1 + muj)
        exp_kappa_t = np.exp(kappa * T)
        lambdaj_t = lambda_ * T
        lambdaj_vj = lambda_ * vj
        theta_lambdaj_vj = (4 * theta + lambdaj_vj * (4 + vj))

        Kurtrt = (3 * sigma2 * (- 4 * kappa * rho + sigma)**2 * theta * (sigma2 + 2 * kappa * theta) + 12 * exp_kappa_t * sigma * theta * (7 * sigma**5 - 
            kappa * sigma**3 * (56 * rho*sigma - 5 * sigma2 * T + theta) + kappa2 * sigma2 * (- 40 * rho * sigma2 * T + sigma**3 * T**2 + 
            8 * rho * theta + sigma * (24 + 136 * rho**2 + T * theta)) - 4 * kappa3 * sigma * (24 * rho**3 * sigma - 3 * sigma2 * T + 4 * rho**2 * (- 6 * sigma2 * T + theta) + 
            2 * rho * sigma * (12 + sigma2 * T**2 + T * theta)) - 4 * kappa**5 * rho * T * (- 8 * rho * sigma + 4 * rho**2 * sigma2 * T + theta_lambdaj_vj) + 
            kappa4 * sigma * (8 - 48 * rho * sigma * T - 64 * rho**3 * sigma * T + 4 * rho**2 * (16 + 5 * sigma2 * T**2 + 4 * T * theta) + T * theta_lambdaj_vj)) + 
            exp_kappa_t**2 * (- 87 * sigma**6 * theta + 6 * kappa * sigma**4 * theta * (116 * rho * sigma + 5 * sigma2 * T + theta) + 6 * kappa3 * sigma2 * theta * (192 * rho**3 * sigma + 
            16 * rho**2 * (6 * sigma2 * T + theta) + 16 * rho * sigma * (12 + T * theta) + sigma2 * T * (24 + T*theta)) - 12 * kappa2 * sigma**3 * theta * (20 * rho * sigma2 * T + 
            4 * rho * theta + sigma * (24 + 140 * rho**2 + T * theta)) - 48 * kappa**6 * rho * sigma * T**2 * theta * theta_lambdaj_vj - 
            12 * kappa4 * sigma2 * theta * (8 + 32 * rho**3 * sigma * T + 16 * rho**2 * (4 + T * theta) + 4 * rho * sigma * T * (12 + T * theta) + 
            T * theta_lambdaj_vj) + 2 * kappa7 * T * (lambda_ * vj**2 * (48 + 24 * vj + vj**2) + 3 * T * theta_lambdaj_vj**2) + 
            12 * kappa**5 * sigma * T * theta * (4 * rho * theta_lambdaj_vj + sigma * (8 + 8 * rho**2 * (4 + T * theta) + T * theta_lambdaj_vj))) - 
            16 * exp_kappa_t * kappa4 * lambdaj_t * vj * (- 3 * (- 1 + exp_kappa_t) * sigma2 * theta - 12 * exp_kappa_t * kappa2 * rho * sigma * T * theta + 
            3 * kappa * sigma * (4 * (- 1 + exp_kappa_t) * rho + exp_kappa_t * sigma * T) * theta + exp_kappa_t * kappa3 * (vj * (12 + vj) +
            3 * T * theta_lambdaj_vj)) * log_muj + 48 * exp_kappa_t * kappa4 * lambdaj_t * ((1 - exp_kappa_t) * sigma2 * theta - 
            4 * exp_kappa_t * kappa2 * rho * sigma * T * theta + kappa * sigma * (4 * (- 1 + exp_kappa_t) * rho + exp_kappa_t * sigma * T) * theta + 
            exp_kappa_t * kappa3 * (vj * (4 + vj) + T * (4 * theta + lambdaj_vj * (4 + 3 * vj)))) * log_muj**2 - 
            64 * exp_kappa_t**2 * kappa7 * lambdaj_t * (1 + 3 * lambdaj_t) * vj * log_muj**3 + 32 * exp_kappa_t**2 * kappa7 * lambdaj_t * (1 + 3 * lambdaj_t) * log_muj**4) / (2 * exp_kappa_t**2 * kappa7 * (((- 1 + 1 / exp_kappa_t) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + 
            kappa * sigma * ((4 - 4 / exp_kappa_t) * rho + sigma * T) * theta + kappa3 * T * theta_lambdaj_vj) / kappa3 - 4 * lambdaj_t * vj * log_muj + 
            4 * lambdaj_t * log_muj**2)**2)
    
        return Kurtrt

    def Kurtrt(self):
        vj = self.heston_instance.vj
        muj = self.heston_instance.muj
        lambda_ = self.heston_instance.lambda_
        
        if (lambda_ == 0) & (muj == 0) & (vj == 0):
            return self.__Kurtrt_Heston()
        else:
            return self.__Kurtrt_Bates()
        
        