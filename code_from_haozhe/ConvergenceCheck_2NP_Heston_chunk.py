# Heston Moments

import numpy as np
import pandas as pd
from lightning_lite.utilities.seed import seed_everything
import math
from scipy.special import factorial
from scipy.stats import skew,kurtosis
from datetime import datetime
import os

from SimHestonQE import Heston_QE
from Moments import MomentsCIR,getSkFromMoments,getKuFromMoments, MomentsBates
from RealizedMomentsEstimator_MonthOverlap import rMoments, rMoments_nc

# # Check theoretical moments 
print('*****************************************************Theoretical Moments*****************************************************************')
Evp = MomentsBates(mu = 0.19/2, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.1, conditional=False)   # v0 = 0.1
Evp_nc = MomentsBates(mu = 0.19/2, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.1, conditional=False, nc=True)   # v0 = 0.1
print(Evp)
print(Evp_nc)
print('\n'*1)
# # Check realized moments from QE
print('*****************************************************QE Realized Moments*****************************************************************')

def Chunk_RealizedMomentsEstimator(chunk):
    
    time_points = 11 * 22 * 79 * 12       # to match 11 year high frequency data
    T = 11          # simulate 11 year data
    S0 = 100
    paths = chunk
    QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0.19/2, rho=-0.7, T=T, n=time_points, M=paths)[:,1:]    # the first point is lnS0 which is excluded
    QE_process_transform = QE_process.T         # need to transform it to set the index

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

    # set the first 10 years as burnin
    df_logreturn = df_logreturn.iloc[10 * 22 * 79 * 12:]

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
        rm.append(rMoments(df_logreturn[column], method=technique, months_overlap=6, m1zero=True, ret_nc_mom=True).to_numpy())
    rm = np.squeeze(np.array(rm))
    rm = np.mean(rm, axis=1)      # average along the time series, so get average realized mean, var, skew, kurt w.r.t. each path

    return rm

def Chunk_RealizedMomentsEstimator_2(chunk):
    
    time_points = 11 * 22 * 79 * 12       # to match 11 year high frequency data
    T = 11          # simulate 11 year data
    S0 = 100
    paths = chunk
    QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0.19/2, rho=-0.7, T=T, n=time_points, M=paths)[:,1:]    # the first point is lnS0 which is excluded
    QE_process_transform = QE_process.T         # need to transform it to set the index

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

    # set the first 10 years as burnin
    df_logreturn = df_logreturn.iloc[10 * 22 * 79 * 12:]

    # Define constants for the realized moments
    RM_ACJV = 1  # Amaya et al, 2015
    RM_CL = 2    # Choe and Lee, 2014
    RM_NP = 3    # Neuberger and Payne, 2018
    RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
    RM_NP_return = 5    # Neuberger and Payne, 2020, in return form


    # Compute realized moments using NP price difference method
    technique_2 = RM_NP
    rm_2 = []
    for column in df_logreturn.columns:
        rm_2.append(rMoments(df_logreturn[column], method=technique_2, months_overlap=6, m1zero=True, ret_nc_mom=True).to_numpy())
    rm_2 = np.squeeze(np.array(rm_2))
    rm_2 = np.mean(rm_2, axis=1)      # average along the time series, so get average realized mean, var, skew, kurt w.r.t. each path

    return rm_2


seed_everything(33)
rm = []
for i in range(400):
    processed_chunk = Chunk_RealizedMomentsEstimator(500)
    rm.append(processed_chunk)
rm = np.vstack(rm)
#print(rm)
print(rm.shape)            # the shape should be (no. of chunks * chunk size, 6) , i.e., (200000, 6)

rm_2 = []
for i in range(400):
    processed_chunk_2 = Chunk_RealizedMomentsEstimator_2(500)
    rm_2.append(processed_chunk_2)
rm_2 = np.vstack(rm_2)

print(rm_2.shape)            # the shape should be (no. of chunks * chunk size, 6) , i.e., (200000, 6)

# save the data
import pickle
with open('NP_return_Heston_1y_6m_kappa3_10yburnout_mu0095_v0019.pickle', 'wb') as file:
    pickle.dump(rm, file)

with open('NP_diff_Heston_1y_6m_kappa3_10yburnout_mu0095_v0019.pickle', 'wb') as file:
    pickle.dump(rm_2, file)

print('*****************************************************Convergence Check is done*****************************************************************')

