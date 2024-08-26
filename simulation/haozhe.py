import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments

np.random.seed(0)

time_points = 3 * 12 * 22 * 79 # 3 years with 12 months with 22 trading days with 79 prices (5 minute intervals from 9:30 am to 4pm and inital price
T = 3
S0 = 100
paths = 20

QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, n=time_points, M=paths) # the first point is lnS0
QE_process_transform = QE_process[:,1:].T # need to transform it to set the index

# Create a DatetimeIndex for every 5 minutes during trading hours
start_date = '2014-07-01'
end_date = '2026-07-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
time_range = pd.date_range(start='09:30', end='16:05', freq='5min')[:-1]  # Trading hours making it actually end at 16:00

# Create a full DatetimeIndex for all trading days and intraday times
datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

# Ensure the datetime_index length matches the data length
datetime_index = datetime_index[:time_points]

# Create the DataFrame
df = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(paths)])
firstday = df.iloc[0]
df_logreturn = df.diff() # compute the log returns
df_logreturn.iloc[0] = firstday - np.log(S0) # the first log-return is compared to the S0

# print(df)

# set the first 2 years and 10 months as burnin
# we want to estimate unconditional moments, so we need to remove the burnin period because inital parameters are not unconditional
df_logreturn = df_logreturn.iloc[2 * 22 * 79 * 12 + 22 * 79 * 10:]

# print(df_logreturn)

# Define constants for the realized moments
RM_ACJV = 1  # Amaya et al, 2015
RM_CL = 2    # Choe and Lee, 2014
RM_NP = 3    # Neuberger and Payne, 2018
RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

# Compute realized moments using NP return method
technique = RM_CL
rm = []
for column in df_logreturn.columns:
    rm.append(rMoments(df_logreturn[column], method=technique, days_aggregate=22, m1zero=True, ret_nc_mom=True).to_numpy())
rm = np.squeeze(np.array(rm))

print(rm.shape)