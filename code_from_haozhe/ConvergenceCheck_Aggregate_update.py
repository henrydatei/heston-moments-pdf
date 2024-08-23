# Heston Moments
import torch 
import numpy as np
import pandas as pd
from lightning_lite.utilities.seed import seed_everything
import math
from scipy.special import factorial
from scipy.stats import skew,kurtosis
from datetime import datetime
import os

from SimHestonQE import Heston_QE
from Moments_list import MomentsCIR,getSkFromMoments,getKuFromMoments, MomentsBates
from RealizedMomentsEstimator_Aggregate_update import rMoments, rMoments_nc

# # Check theoretical moments 
print('*****************************************************Theoretical Moments*****************************************************************')
Evp = MomentsBates(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)   # v0 = 0.1
Evp_nc = MomentsBates(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False, nc=True)   # v0 = 0.1
print(Evp)
print(Evp_nc)
print('\n'*1)


# # Check realized moments from QE
print('*****************************************************QE Realized Moments*****************************************************************')

def Chunk_RealizedMomentsEstimator(chunk):
    
    time_points = 3 * 22 * 79 * 12       # to match 3 year high frequency data
    T = 3          # simulate 3 year data
    S0 = 100
    paths = chunk
    QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, n=time_points, M=paths)    # the first point is lnS0
    QE_process_transform = QE_process[:,1:].T         # need to transform it to set the index

    # Create a DatetimeIndex for every 5 minutes during trading hours
    start_date = '2014-07-01'
    end_date = '2026-07-01'
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    time_range = pd.date_range(start='09:30', end='16:05', freq='5T')[:-1]  # Trading hours making it actually end at 16:00

    # Create a full DatetimeIndex for all trading days and intraday times
    datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

    # Ensure the datetime_index length matches the data length
    datetime_index = datetime_index[:time_points]

    # Create the DataFrame
    df = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(paths)])
    firstday = df.iloc[0]
    df_logreturn = df.diff()        # compute the log returns
    df_logreturn.iloc[0] = firstday - np.log(S0)    # the first log-return is compared to the S0

    # set the first 2 years as burnin
    df_logreturn = df_logreturn.iloc[2 * 22 * 79 * 12 + 22 * 79 * 10:]

    # Define constants for the realized moments
    RM_ACJV = 1  # Amaya et al, 2015
    RM_CL = 2    # Choe and Lee, 2014
    RM_NP = 3    # Neuberger and Payne, 2018
    RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
    RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

    # Compute realized moments using NP return method
    technique = RM_NP_return
    rm = []
    for column in df_logreturn.columns:
        rm.append(rMoments(df_logreturn[column], method=technique, days_aggregate=22, m1zero=True, ret_nc_mom=True).to_numpy())
    rm = np.squeeze(np.array(rm))

    return rm


seed_everything(333)
rm = []
for i in range(400):
    processed_rm = Chunk_RealizedMomentsEstimator(500)
    rm.append(processed_rm)
rm = np.vstack(rm)
#print(rm)
print(rm.shape)            # the shape should be (no. of chunks * chunk size, 6) , i.e., (200000, 6)


# save the data
import pickle
with open('NP_return_22d_kappa3_2yburnout_mu0_v0019_update.pickle', 'wb') as file:
    pickle.dump(rm, file)

print('*****************************************************Data Saving is done*****************************************************************')