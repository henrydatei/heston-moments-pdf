import numpy as np
import yaml

"""
RHeston: Enhanced Stochastic Volatility Model with Randomized Parameter Sampling

The `RHeston` class extends the Heston stochastic volatility model by introducing parameter bounds and randomized sampling functionality. It is designed for simulation purposes, providing methods to sample model parameters while ensuring compliance with the Feller condition.

Key Features:
-------------
- Automatic correction of parameter bounds with the `do_correct_limits` flag.
- Randomized parameter sampling methods, including those ensuring the Feller condition.
- Extended functionality for generating time series data using the randomized Quadratic Exponential (QE) method.

Attributes:
-----------
- do_correct_limits (bool): Enables automatic correction of invalid parameter bounds.
- is_feller_satisfied (bool): Ensures all sampled parameters satisfy the Feller condition.
- theta_l, theta_u (float): Bounds for the long-term variance parameter `theta`.
- sigma_l, sigma_u (float): Bounds for the volatility parameter `sigma`.
- rho_l, rho_u (float): Bounds for the correlation parameter `rho`.
- feller_l (float): Lower bound for the Feller condition.  HJ: this is actually not a bound for Feller but a scale parameter of the exponential distribution to gneerate Feller values.

Methods:
--------
- set_sigma_limits(sigma_l, sigma_u):
    Validates and sets bounds for the volatility parameter `sigma`. Automatically corrects invalid bounds if `do_correct_limits` is enabled.

- set_theta_limits(theta_l, theta_u):
    Validates and sets bounds for the long-term variance parameter `theta`. Automatically corrects invalid bounds if enabled.

- set_rho_limits(rho_l, rho_u):
    Validates and sets bounds for the correlation parameter `rho`. Automatically corrects invalid bounds if enabled.

- set_feller_limit(feller_l):
    Sets the lower bound for the Feller condition. Corrects invalid bounds if enabled. HJ: this is actually not a bound for Feller but a scale parameter of the exponential distribution to gneerate Feller values.

- set_parameter_bounds(theta_l=None, theta_u=None, sigma_l=None, sigma_u=None, rho_l=None, rho_u=None, feller_l=None):
    A comprehensive method for setting all parameter bounds simultaneously.

- get_random_param_set():
    Samples a random set of parameters within specified bounds, without ensuring compliance with the Feller condition.

- get_random_param_set_fellercond():
    Samples a random set of parameters that satisfies the Feller condition, with repeated sampling if necessary.

- rPar_rQE(n, M=1, v0=None, v=None, Z_v=None, gam1=0.5, gam2=0.5):
    Generates time series data using the randomized QE method, sampling parameters based on the current settings of the `is_feller_satisfied` flag.

Usage:
------
The `RHeston` class is ideal for scenarios where parameter uncertainty needs to be incorporated into model simulations, such as in Monte Carlo studies or stress testing. By enforcing the Feller condition, the class ensures the mathematical validity of the simulated stochastic processes.

Example:
--------
from RHeston import RHeston

# Initialize the model
model = RHeston(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, S0=100, T=1)

# Set custom parameter bounds
model.set_parameter_bounds(theta_l=0.01, theta_u=0.1, sigma_l=0.2, sigma_u=1.0, rho_l=-0.9, rho_u=0.9, feller_l=2)

# Generate a random parameter set satisfying the Feller condition
model.get_random_param_set_fellercond()

# Simulate time series data
data = model.rPar_rQE(n=100, M=5)
"""

from Heston import Heston

### Child class, that allow for random sampling of the parameters
class RHeston(Heston):
    def __init__(self, kappa=1, theta=0.5, sigma=0.5, rho=0, lambda_=0, muj=0, vj=0, mu=0, S0=100, T = 1, yaml_path = None):
        """
        Initializes the RHeston class with default parameters and bounds.

        Parameters:
        -----------
        kappa : float
            Rate of mean reversion for variance.
        theta : float
            Long-term variance.
        sigma : float
            Volatility of variance.
        rho : float
            Correlation between the asset and variance.
        lambda_ : float
            Jump intensity (default is 0).
        muj : float
            Mean of jump size (default is 0).
        vj : float
            Variance of jump size (default is 0).
        mu : float
            Drift rate (default is 0).
        S0 : float
            Initial asset price (default is 100).
        T : float
            Time interval to consider a return (default is 1).

        Notes:
        ------
        Sets default bounds for parameters such as `theta`, `sigma`, `rho`, 
        and the Feller condition.
        """

        super().__init__(kappa, theta, sigma, rho, lambda_, muj, vj, mu, S0, T, yaml_path=yaml_path)

        if yaml_path:
            self._load_rHeston_parameters_from_yaml(yaml_path)
        else:
            # Parameter bounds for random sampling
            self.set_parameter_bounds(
                theta_l  = 0.05, 
                theta_u  = 0.5, 
                sigma_l  = 0.1, 
                sigma_u  = 1,            # before it was 10, HJ changed it to be 1.
                rho_l    = -1, 
                rho_u    = 1, 
                feller_l = 0.25          # before it was 4, HJ changed it to be 0.25.
            )
            
            # if triggered: if the limits lie outside natural bounds, they will be set to bounds, if the order is wrong, f.e. sigma_l > sigma_u, the order is reversed.
            self.do_correct_limits = True 
            
            # if the trigger is on, it samples only cases, when Feller condition is satisfied. By triggering it on, no new parameters will be generated. The Feller condition will be controlled by the next generation
            self.is_feller_satisfied = True
        
    def _load_rHeston_parameters_from_yaml(self, yaml_path):
        """
        Load parameters from a YAML file.

        Parameters:
        -----------
        yaml_path : str
            Path to the YAML file containing parameters.
        """
        
        with open(yaml_path, 'r') as file:
            params = yaml.safe_load(file)            
        
        # Access nested Heston group
        rheston_params = params.get('rHeston', {})
        
        self.set_parameter_bounds(
            theta_l  = rheston_params.get('theta_l', 0.05), 
            theta_u  = rheston_params.get('theta_u', 0.5), 
            sigma_l  = rheston_params.get('sigma_l', 0.1), 
            sigma_u  = rheston_params.get('sigma_u', 1),  # before it was 10, HJ changed it to be 1. 
            rho_l    = rheston_params.get('rho_l', -1), 
            rho_u    = rheston_params.get('rho_u', 1), 
            feller_l = rheston_params.get('feller_l', 0.25)
        )
        
        self.do_correct_limits = rheston_params.get('verbose', True) # if do not show any warnings, but silently continue
        # if this is triggered: by setting the wrong parameter, it will not raise error but set the value at the boundary for rho and mirror for the only positive ones (sigma, kappa).
        self.is_feller_satisfied = rheston_params.get('is_feller_satisfied', True)
        # if this is triggered: if kappa is too large, resample from uniform distribution: (0, 500 / self.T)
        
    @property
    def feller_l(self):
        return self._feller_l

    @property
    def theta_l(self):
        return self._theta_l

    @property
    def theta_u(self):
        return self._theta_u

    @property
    def sigma_l(self):
        return self._sigma_l

    @property
    def sigma_u(self):
        return self._sigma_u

    @property
    def rho_l(self):
        return self._rho_l

    @property
    def rho_u(self):
        return self._rho_u

    def set_sigma_limits(self, sigma_l, sigma_u):
        """
        Validates and sets limits for the volatility parameter `sigma`.
        
        Automatically corrects limits if `do_correct_limits` is enabled.

        Parameters:
        -----------
        sigma_l : float
            Lower limit for `sigma`.
        sigma_u : float
            Upper limit for `sigma`.

        Raises:
        -------
        ValueError:
            If invalid limits are provided and corrections are disabled.
        """
        
        if sigma_l > 0 and sigma_u > sigma_l:
            self._sigma_l, self._sigma_u = sigma_l, sigma_u
        elif self.do_correct_limits:
            if sigma_l < 0 and sigma_u < 0:
                # Both sigma_l and sigma_u are negative
                sigma_l, sigma_u = np.abs(sigma_l), np.abs(sigma_u)
                self._sigma_l, self._sigma_u = min(sigma_l, sigma_u), max(sigma_l, sigma_u)
            elif sigma_l <= 0:
                # Only sigma_l is negative
                self._sigma_l = 0.01
                self._sigma_u = sigma_u
            elif sigma_u <= 0:
                # sigma_l is positive and sigma_u is negative
                self._sigma_l = 0.01
                self._sigma_u = sigma_l
            elif sigma_l > 0 and sigma_u > 0 and sigma_u < sigma_l:
                # Both are positive, but sigma_u < sigma_l
                self._sigma_l, self._sigma_u = sigma_u, sigma_l
            else:
                # If no corrections are applicable
                raise ValueError("Unable to correct sigma limits.")
            if not self.verbose: 
                print(f"Invalid sigma limits. We corrected them: sigma_l = {self._sigma_l}, sigma_u = {self._sigma_u}. If you do not want this set 'do_correct_limits' to false")
        else:
            raise ValueError("Invalid sigma limits: sigma_l must be > 0 and sigma_u > sigma_l")

    def set_theta_limits(self, theta_l, theta_u):
        """
        Validates and sets limits for the long-term variance parameter `theta`.

        Parameters:
        -----------
        theta_l : float
            Lower limit for `theta`.
        theta_u : float
            Upper limit for `theta`.

        Raises:
        -------
        ValueError:
            If invalid limits are provided and corrections are disabled.
        """
        
        if 0 <= theta_l < theta_u:
            self._theta_l, self._theta_u = theta_l, theta_u
        elif self.do_correct_limits:
            if theta_l < 0 and theta_u < 0:
                # Both theta_l and theta_u are negative
                theta_l, theta_u = np.abs(theta_l), np.abs(theta_u)
                self._theta_l, self._theta_u = min(theta_l, theta_u), max(theta_l, theta_u)
            elif theta_l < 0:
                # Only theta_l is negative
                self._theta_l = 0
                self._theta_u = theta_u
            elif theta_u <= 0:
                # theta_l is positive and theta_u is negative
                self._theta_l = 0
                self._theta_u = theta_l
            elif theta_l > 0 and theta_u > 0 and theta_u < theta_l:
                # Both are positive, but theta_u < theta_l
                self._theta_l, self._theta_u = theta_u, theta_l
            else:
                # If no corrections are applicable
                raise ValueError("Unable to correct theta limits.")
            if not self.verbose: 
                print(f"Invalid theta limits. We corrected them: theta_l = {self._theta_l}, theta_u = {self._theta_u}. If you do not want this set 'do_correct_limits' to false")
        else:
            raise ValueError("Invalid theta limits: 0 <= theta_l < theta_u")

    def set_rho_limits(self, rho_l, rho_u):
        """
        Validates and sets limits for the correlation parameter `rho`.

        Parameters:
        -----------
        rho_l : float
            Lower limit for `rho`.
        rho_u : float
            Upper limit for `rho`.

        Raises:
        -------
        ValueError:
            If invalid limits are provided and corrections are disabled.
        """

        if -1 <= rho_l < rho_u <= 1:
            self._rho_l, self._rho_u = rho_l, rho_u
        elif self.do_correct_limits:
            if (rho_l < -1 and rho_u < -1) or (rho_u > 1 and rho_l > 1) or (rho_u < -1 and rho_l > 1) or (rho_u > 1 and rho_l < -1):
                # Both rho_l and rho_u are outside bounds
                self._rho_l, self._rho_u = -1, 1
            elif rho_l < -1:
                # Only rho_l is out of bounds < -1
                self._rho_l = -1
                self._rho_u = rho_u
            elif rho_u < -1:
                # Only rho_l is out of bounds < -1
                self._rho_l = -1
                self._rho_u = rho_l
            elif rho_u > 1:
                # Only rho_u is out of bounds > 1
                self._rho_l = rho_l
                self._rho_u = 1
            elif rho_l > 1:
                # Only rho_u is out of bounds > 1
                self._rho_l = rho_u
                self._rho_u = 1
            elif rho_l >= -1 and rho_u <= 1 and rho_u < rho_l:
                # Both are in bounds, but rho_u < rho_l
                self._rho_l, self._rho_u = rho_u, rho_l
            else:
                # If no corrections are applicable
                raise ValueError("Unable to correct rho limits.")
            if not self.verbose: 
                print(f"Invalid rho limits. We corrected them: rho_l = {self._rho_l}, rho_u = {self._rho_u}. If you do not want this set 'do_correct_limits' to false")
        else:
            raise ValueError("Invalid rho limits: -1 <= rho_l < rho_u <= 1")

    def set_feller_limit(self, feller_l):
        """
        Validates and sets the lower bound for the Feller condition. HJ: this is actually not a bound for Feller but a scale parameter of the exponential distribution to gneerate Feller values.

        Parameters:
        -----------
        feller_l : float
            Minimum value to ensure compliance with the Feller condition.

        Raises:
        -------
        ValueError:
            If `feller_l` is less than or equal to zero and corrections are disabled.
        """

        if feller_l > 0:
            self._feller_l = feller_l
        elif self.do_correct_limits:
            self._feller_l = np.abs(feller_l)
            if not self.verbose: 
                print(f"Invalid feller_l. We corrected them: feller_l = {self._feller_l}. If you do not want this set 'do_correct_limits' to false")
        else:
            raise ValueError("feller_l must be > 0")

    def set_parameter_bounds(self, theta_l=None, theta_u=None, sigma_l=None, sigma_u=None, rho_l=None, rho_u=None, feller_l=None):
        """
        Sets parameter bounds for all relevant parameters.

        Parameters:
        -----------
        theta_l, theta_u : float
            Bounds for the long-term variance parameter.
        sigma_l, sigma_u : float
            Bounds for the volatility parameter.
        rho_l, rho_u : float
            Bounds for the correlation parameter.
        feller_l : float
            Lower bound for the Feller condition.
        """

        self.set_feller_limit(feller_l)
        self.set_theta_limits(theta_l, theta_u)
        self.set_sigma_limits(sigma_l, sigma_u)
        self.set_rho_limits(rho_l, rho_u)
        
    # get the random set of parameters
    def __get_random_param_set_nofeller_control(self):
        """
        Generates a random set of parameters within the specified bounds.

        Notes:
        ------
        Does not guarantee compliance with the Feller condition. Use 
        `get_random_param_set_fellercond` for this.
        """
        
        # feller = rexp(num_random, 4) + 1        # in Python, the argument is the scale parameter, which is the inverse of rate parameter in R
        # theta = runif(num_random, 0.05, 0.5)
        # sigma = runif(num_random, 0.1, 10)
        # rho = runif(num_random, -0.99, 0.99)
        ### I do not touch the Bates parameters
        # lambda_ = 0.
        # muj = 0.
        # vj = 0.
        # mu = 0.
        # S0 = 100
        # time_points = 3 * 22 * 79 * 12       # to match 3 year high frequency data
        # T = 3                                # simulate 3 year data        
        feller = np.random.exponential(self.feller_l, 1).item() + 1 # generate Feller condition number (not in self, since it will be by everty call updated and computed for all kappa, sigma and theta)
        self.theta = np.random.uniform(self.theta_l, self.theta_u, 1).item()
        self.sigma = np.random.uniform(self.sigma_l, self.sigma_u, 1).item()
        self.rho = np.random.uniform(self.rho_l, self.rho_u, 1).item()
        self.kappa = feller * self.sigma**2 / self.theta / 2          # imply the kappa which makes the Feller condition hold
        self.mu = self.theta / 2
        
    # obtain random values that satisfy Feller condition
    def __get_random_param_set_fellercond(self):
        """
        Generates a random set of parameters that satisfy the Feller condition.

        Notes:
        ------
        May involve repeated sampling if initial parameters do not satisfy the 
        Feller condition.
        """
        
        self.__get_random_param_set_nofeller_control()
        i = 0
        while self.feller < 1:                           # self.feller is explicitly assigned in Heston class.
            i = i + 1
            self.__get_random_param_set_nofeller_control()                
        if i != 0: 
            if not self.verbose: 
                print(f"Warning: We repeated {i} times sampling to get the set of parameters that satisfies the Feller condition")
        
    def get_random_param_set(self):
        """
        Generates a random set of parameters, ensuring compliance with the Feller condition
        and that Skewrt() and Kurtrt() produce valid (non-inf, non-NaN) values.
        """
        max_attempts = 10  # Limit the number of attempts to prevent infinite loops
        for attempt in range(max_attempts):
            # Generate a random parameter set
            if self.is_feller_satisfied:
                self.__get_random_param_set_fellercond()
            else:
                self.__get_random_param_set_nofeller_control()

            # Validate Skewrt() and Kurtrt()
            try:
                skewness = self.moments.Skewrt()
                kurtosis = self.moments.Kurtrt()
                if np.isfinite(skewness) and np.isfinite(kurtosis):
                    # If valid, break the loop and keep the parameters
                    if not self.verbose:
                        print(f"Valid parameter set found after {attempt + 1} attempts.")
                    return  # Stop checking; current parameters are valid
            except Exception as e:
                # Log the exception if verbose mode is enabled
                if not self.verbose:
                    print(f"Error evaluating moments: {e}")

        # If no valid parameters are found after max_attempts, raise an error
        raise ValueError("Unable to generate valid parameters after multiple attempts.")
        
