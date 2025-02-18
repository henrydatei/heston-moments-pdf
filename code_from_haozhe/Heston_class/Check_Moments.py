import torch 
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F 
from torch.optim import Adam 

import pytorch_lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

#from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning import seed_everything
from SimHestonQE import Heston_QE
from Moments_list import MomentsCIR,getSkFromMoments,getKuFromMoments, MomentsBates
from Moments_Skewness import MomentsBates_Skewness
from Moments_Kurtosis import MomentsBates_Kurtosis
import numpy as np
import pandas as pd
from datetime import datetime
import random

# import Heston class
from Heston import Heston
from moments import Moments

start_time = datetime.now()
print('*****************************************************Data Loading Starts*****************************************************************')

batch_size = 64
test_batch_size = 128

# Initialize empty lists to store input sequences and labels
X = []  # Input sequences
y = []  # Corresponding labels
num_simulation = 2       #100
num_random = 3           #900
seed_everything(seed=33)
#random.seed(33)

# setting parameters
#mu = 0
S0 = 100
time_points = 3 * 22 * 79 * 12       # to match 3 year high frequency data
T = 3                                # simulate 3 year data
feller, theta, sigma, rho = np.random.exponential(0.25, num_random) + 1, np.random.uniform(0.05, 0.5, num_random), np.random.uniform(0.1, 10, num_random), np.random.uniform(-0.99, 0.99, num_random)
kappa = feller * sigma**2 / theta / 2          # imply the kappa which makes the Feller condition hold
# Create input sequences and labels
for i in range(num_random):    # go through each random set of parameters
    print(f"Iteration {i} params: theta={theta[i]}, sigma={sigma[i]}, rho={rho[i]}, kappa={kappa[i]}")
    true_cumulants = MomentsBates(mu = theta[i]/2, kappa = kappa[i], theta = theta[i], sigma = sigma[i], rho=rho[i], lambdaj=0, muj=0, vj=0, t = 1/12, v0 = theta[i], conditional=False)
    skewness = MomentsBates_Skewness(mu = theta[i]/2, kappa = kappa[i], theta = theta[i], sigma = sigma[i], rho=rho[i], lambdaj=0, muj=0, vj=0, t = 1/12, v0 = theta[i], conditional=False)
    kurtosis = MomentsBates_Kurtosis(mu = theta[i]/2, kappa = kappa[i], theta = theta[i], sigma = sigma[i], rho=rho[i], lambdaj=0, muj=0, vj=0, t = 1/12, v0 = theta[i], conditional=False)
    print("true cumulants are:", true_cumulants)
    print("true skewness is:", skewness)
    print("true kurtosis is:", kurtosis)

    heston = Heston(kappa = kappa[i], theta = theta[i], sigma = sigma[i], rho=rho[i], lambda_=0, muj=0, vj=0, mu = theta[i]/2, S0=100, T = 1/12)
    class_var = heston.moments.Varrt()
    print("true class variance is:", class_var)

    class_skewness = heston.moments.Skewrt()
    print("true class skewness is:", class_skewness)

    class_kurt = heston.moments.Kurtrt()
    print("true class kurtosis is:", class_kurt)

    print(heston.get_params())
    #class_theta = heston.theta
    #print("true class theta is:", class_theta)

