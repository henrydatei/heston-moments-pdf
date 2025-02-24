import pytorch_lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data
import torch 
from pytorch_lightning import seed_everything
import numpy as np
import pandas as pd
from datetime import datetime
from Network import HestonNet
from rHeston import RHeston
from CNNinput import Convert_to_CNN_input
from ImplementNP import NP_Estimator_MonthlyOverlap
start_time = datetime.now()

# Load the saved state dict
state_dict = torch.load("Fly_CNN_1.pt", map_location=torch.device("cpu"))

# Remove the "network." prefix
new_state_dict = {key.replace("network.", ""): value for key, value in state_dict.items()}

# Load the CNN model
model_CNN = HestonNet() 
model_CNN.load_state_dict(new_state_dict)
# Set model to evaluation mode
model_CNN.eval()

print('*****************************************************Model Loading is done*****************************************************************')
print('\n'*1)

print('*****************************************************Data Loading Starts*****************************************************************')
# Initialize empty lists to store input sequences and labels
rCumu, rSkew, Skew, CNN_output = ([] for i in range(4))
num_simulation = 50   #100
num_random = 1000       #900
seed_everything(seed=33)

# setting parameters
S0 = 10
time_points = 3 * 22 * 79 * 12       # to match 3 year high frequency data
burnin = 2 * 22 * 79 * 12 + 22 * 79 * 0   # 2 years burnin
T = 3                                # simulate 3 year data

# Initialize RHeston model instance
heston_model = RHeston(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, mu=0.02, S0=10, T=1/12)   #here T represents the time interval of the log-return

# Create input sequences and labels
for i in range(num_random):    # go through each random set of parameters
    heston_model.get_random_param_set()    # update Heston parameter set
    print(f"Iteration {i} params = {heston_model.get_params()}")

    skewness = heston_model.moments.Skewrt()         # derive the true skewness
    print(f"Iteration {i} true skewness = {skewness}")

    QE_process = heston_model.sampling.rQE(n=time_points, TT=T, M=num_simulation)    # here TT represents total time horizon for simulation    
    QE_process = np.diff(QE_process)                    # to get the difference, i.e., log-returns
    QE_process_cut = QE_process[:, burnin:]         # remove the burnin part

    # Transform log-returns into CNN format
    cnn_input = Convert_to_CNN_input(QE_process_cut, order=3)
    #print(cnn_input.shape)

    # Compute CNN output and append
    input_tensor = torch.tensor(cnn_input, dtype=torch.float32)   # convert the CNN input data into tensor type
    CNN_predict = model_CNN(input_tensor).detach().cpu().numpy()[:, 0].reshape((-1, 1))
    CNN_output.append(CNN_predict)

    # NP realized cumulants
    rc = NP_Estimator_MonthlyOverlap(QE_process=QE_process_cut, time_points=time_points, burnin=burnin)
    
    rCumu.append(rc)
    #rSkew.append(rk)
    Skew.append(np.tile(skewness, (QE_process_cut.shape[0], 1)))           # repeat the desired output for no. of simulations to match the size

# rCumu, Skew = np.vstack(rCumu), np.vstack(Skew)      # stack the input and label vertically to have the size (no. of simulations * no. of random, 11, no. of features) for input and (no. of simulations * no. of random, 1) for label
# # rCumu, Skew, CNN_output = np.array(rCumu), np.array(Skew), np.array(CNN_output)
# print(rCumu.shape)           # shape: (no. of random sets, no. of simulations, 11, 4)
# print(Skew.shape)            # shape: (no. of random sets, no. of simulations, 1)
# print(np.array(CNN_output).shape)      # shape: (no. of random sets, no. of simulations, 1)

# save the data
import pickle
with open('rCumulants_CNN_2mOverlap.pickle', 'wb') as file:
    pickle.dump(rCumu, file)

with open('CNN_skew_2mOverlap.pickle', 'wb') as file:
    pickle.dump(CNN_output, file)

# with open('rSkewness_22d.pickle', 'wb') as file:
#     pickle.dump(rSkew, file)

with open('TrueSkewness_2mOverlap.pickle', 'wb') as file:
    pickle.dump(Skew, file)

print('*****************************************************Data Loading is done*****************************************************************')
end_time = datetime.now()
print(f'Computation Duration is: {end_time - start_time}')