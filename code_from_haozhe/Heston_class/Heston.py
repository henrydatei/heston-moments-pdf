import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml

class Heston:
    def __init__(self, kappa=1, theta=0.5, sigma=0.5, rho=0, lambda_=0, muj=0, vj=0, mu=0, S0=100, T = 1, yaml_path = None):
        if yaml_path:
            self._load_parameters_from_yaml(yaml_path)
        else:
            self._kappa = kappa
            self._theta = theta
            self._sigma = sigma
            self._rho = rho
            self._lambda = lambda_
            self._muj = muj
            self._vj = vj
            self._mu = mu
            self._S0 = S0 # starting price
            self._T = T # time interval to consider a return
            self.verbose = True # if do not show any warnings, but silently continue
            
            # if this is triggered: by setting the wrong parameter, it will not raise error but set the value at the boundary for rho and mirror for the only positive ones (sigma, kappa).
            self.if_wrong_param_set_in_limit = True
            
            # if this is triggered: if kappa is too large, resample from uniform distribution: (0, 500 / self.T)
            self.if_kappa_large_resample = True

            # trigger, if any parameter has been changed, then many things are to be recomputed
        self.parameter_changed = False
        
        #self.np = self._load_numpy() # load numpy        
        
        self.moments = self._create_moments()  # Private method to create Moments instance
        self.sampling = self._create_sampling()  # Private method to create Sampling instance

    def _load_parameters_from_yaml(self, yaml_path):
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
        heston_params = params.get('Heston', {})

        # Initialize with JSON parameters or fall back to default if missing
        self._kappa = heston_params.get('kappa', 1)
        self._theta = heston_params.get('theta', 0.5)
        self._sigma = heston_params.get('sigma', 0.5)
        self._rho = heston_params.get('rho', 0)
        self._lambda = heston_params.get('lambda', 0)
        self._muj = heston_params.get('muj', 0)
        self._vj = heston_params.get('vj', 0)
        self._mu = heston_params.get('mu', 0)
        self._S0 = heston_params.get('S0', 100)
        self._T = heston_params.get('T', 1)
        
        self.verbose = heston_params.get('verbose', True) # if do not show any warnings, but silently continue
        # if this is triggered: by setting the wrong parameter, it will not raise error but set the value at the boundary for rho and mirror for the only positive ones (sigma, kappa).
        self.if_wrong_param_set_in_limit = heston_params.get('if_wrong_param_set_in_limit', True)
        # if this is triggered: if kappa is too large, resample from uniform distribution: (0, 500 / self.T)
        self.if_kappa_large_resample = heston_params.get('if_kappa_large_resample', True)
    
        
    # def _load_numpy(self):
    #     try:
    #         import numpy as np
    #         return np
    #     except ImportError as e:
    #         print(f"Error loading numpy: {e}")
    #         return None                

    def _create_moments(self):
        from moments import Moments
        #print(f"Imported Moments: {Moments}")
        return Moments(self)  # Create an instance of Moments
    
    def _create_sampling(self):
        from sampling import Sampling
        #print(f"Imported Sampling: {Sampling}")
        return Sampling(self)  # Create an instance of Sampling

    def __str__(self):
        return "Heston Model: " + self.print_params()

    @property
    def feller(self):
        # Compute the Feller number based on kappa, theta, and sigma 
        return 2 * self.kappa * self.theta / (self.sigma ** 2)
    
    #### Time interval
    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value > 0:
            if((value * self.kappa) < 250): # in the kurtosis one needs to compute exp(2 * kappa * T), if too large, it cannot be computed # updated self.T to self.kappa
                self._T = value
                self.parameter_changed = True
            else:
                self._T = value
                if(self.if_kappa_large_resample):
                    self.kappa = np.random.uniform(0.01, 250 / value, 1).item()
                    self.parameter_changed = True
                    if not self.verbose: 
                        print(f"Warning: T (time interval) was too large for the given kappa. Kappa has been resampled.")
                else:
                    self.kappa = 250 / self.T
                    self.parameter_changed = True
                    if not self.verbose: 
                        print(f"Warning: T (time interval) was too large for the given kappa. Kappa is capped it by {self.kappa}.")
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: T (time interval) must be positive. Taking the absolute value.")
            self._T = abs(value)
            self.parameter_changed = True
        else:
            raise ValueError("T (time interval) must be positive")

    #### kappa - mean reversion
    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        if value > 0:
            if((value * self.T) < 250): # in the kurtosis one needs to compute exp(2 * kappa * T), if too large, it cannot be computed
                self._kappa = value
                self.parameter_changed = True
            else:
                if(self.if_kappa_large_resample):
                    self._kappa = np.random.uniform(0.01, 250 / self.T, 1).item()
                    self.parameter_changed = True
                    if not self.verbose: 
                        print(f"Warning: kappa (mean reversion) was too large. it has been resampled.")
                else:
                    self._kappa = 250 / self.T
                    self.parameter_changed = True
                    if not self.verbose: 
                        print(f"Warning: kappa (mean reversion) was too large. Cap it by {self.kappa}.")
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: kappa (mean reversion) must be positive. Taking the absolute value.")
            self._kappa = abs(value)
            self.parameter_changed = True
        else:
            raise ValueError("kappa (mean reversion) must be positive")

    #### theta - expected vola
    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self.parameter_changed = True

    #### sigma - vola of vola
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value > 0:
            self._sigma = value
            self.parameter_changed = True
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: sigma (vola of vola) must be positive. Taking the absolute value.")
            self._sigma = abs(value)
            self.parameter_changed = True            
        else:
            raise ValueError("sigma (vola of vola) must be positive")

    #### rho - vola correlation between noises
    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        if -1 <= value <= 1:
            self._rho = value
            self.parameter_changed = True
        else:
            raise ValueError("rho (correlation) must be between -1 and 1")

    #### lambda - jump intensity
    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        if value > 0:
            self._lambda = value
            self.parameter_changed = True
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: lambda (jumps intensity) must be positive. Taking the absolute value.")
            self._lambda = abs(value)
            self.parameter_changed = True            
        else:
            raise ValueError("lambda (jumps intensity) must be positive")

    #### muj - jump size expectation
    @property
    def muj(self):
        return self._muj

    @muj.setter
    def muj(self, value):
        self._muj = value
        self.parameter_changed = True

    #### vj - jump size volatility
    @property
    def vj(self):
        return self._vj

    @vj.setter
    def vj(self, value):
        if value > 0:
            self._vj = value
            self.parameter_changed = True
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: vj must be positive. Taking the absolute value.")
            self._vj = abs(value)
            self.parameter_changed = True            
        else:
            raise ValueError("vj must be > 0")

    #### mu - mean of the process
    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self.parameter_changed = True

    #### S0 - starting price
    @property
    def S0(self):
        return self._S0

    @S0.setter
    def S0(self, value):
        if value > 0:
            self._S0 = value
            self.parameter_changed = True
        elif self.if_wrong_param_set_in_limit:
            if not self.verbose: 
                print("Warning: S0 (starting price) must be positive. Taking the absolute value.")
            self._S0 = abs(value)
            self.parameter_changed = True            
        else:
            raise ValueError("S0 (starting price) must be positive")

    def get_params(self, as_dict=True, digits=2):
        """
        Get the model parameters.
        
        Parameters:
        -----------
        as_dict : bool, optional
            If True, returns parameters as a dictionary.
        digits : int, optional
            Number of decimal places for formatting.
        
        Returns:
        --------
        str or dict
            Model parameters formatted as a string or dictionary.
        """
        params = {
        "mu": float(round(self.mu, digits)),
        "kappa": float(round(self.kappa, digits)),
        "theta": float(round(self.theta, digits)),
        "sigma": float(round(self.sigma, digits)),
        "rho": float(round(self.rho, digits)),
        "lambda": float(round(self.lambda_, digits)),
        "muj": float(round(self.muj, digits)),
        "vj": float(round(self.vj, digits)),
        "S0": float(round(self.S0, digits)),
        "T": float(round(self.T, digits)),
        }
        if as_dict:
            return params
        return ", ".join([f"{key}={value}" for key, value in params.items()])