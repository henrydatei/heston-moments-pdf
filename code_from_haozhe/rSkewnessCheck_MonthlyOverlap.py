import numpy as np
import pandas as pd
from datetime import datetime
from SimHestonQE import Heston_QE
# from RealizedSkewness_NP_MonthlyOverlap import rCumulants, rSkewness, rMoments
from RealizedMomentsEstimator_MonthOverlap import rMoments
# from Moments_Skewness import MomentsBates_Skewness
from Moments_list import MomentsBates
# from Moments_list_2 import MomentsBates2
from TimeTruncation import align_to_business_days

start_time = datetime.now()
np.random.seed(33)

# # Check theoretical moments 
print('*****************************************************Theoretical Moments*****************************************************************')
# skewness = MomentsBates_Skewness(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False) 
cumulants = MomentsBates(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False, nc=False)  
# print(skewness)
print(cumulants)

# # Check realized moments from QE
print('*****************************************************QE Realized Moments*****************************************************************')

time_points = 15 * 22 * 79 * 12       # to match 3 year high frequency data
burnin = 3 * 22 * 79 * 12
T = 15          # simulate 3 year data
S0 = 100
paths = 1
QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, n=time_points, M=paths)    # the first point is lnS0
QE_process = np.diff(QE_process)                    # to get the difference, i.e., log-returns
QE_process_cut = QE_process[:, burnin:]         # set the first 2 years and 10 months as burnin
QE_process_transform = QE_process_cut.T         # need to transform it to set the index

# Create a DatetimeIndex for every 5 minutes during trading hours
start_date = '2014-07-01'
end_date = '2036-07-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
time_range = pd.date_range(start='09:30', end='16:05', freq='5T')[:-1]  # Trading hours making it actually end at 16:00

# Create a full DatetimeIndex for all trading days and intraday times
datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

# Ensure the datetime_index length matches the data length
time_index = time_points - burnin
datetime_index = datetime_index[:time_index]

# Create the DataFrame
df_logreturn = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process_cut.shape[0])])

# make sure the first point is the first business day and the last point is the last business day of a certain month
cut_df_logreturn = align_to_business_days(df_logreturn)

# Define constants for the realized moments
RM_ACJV = 1  # Amaya et al, 2015
RM_CL = 2    # Choe and Lee, 2014
RM_NP = 3    # Neuberger and Payne, 2018
RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

# Compute realized moments using NP return method
technique = RM_NP_return
rm, rc, rk = ([] for i in range(3))
for column in cut_df_logreturn.columns:
    rm.append(rMoments(cut_df_logreturn[column], method=technique, months_overlap=6).to_numpy())
    # rc.append(rCumulants(cut_df_logreturn[column], method=technique, months_overlap=6).to_numpy())
    # rk.append(rSkewness(cut_df_logreturn[column], method=technique, months_overlap=6).to_numpy())

#rm, rc = np.squeeze(np.array(rm)), np.squeeze(np.array(rc))           # no need to squeeze rk because it has one element which has been automatically squeezed by np
rm = np.array(rm)
# rc = np.array(rc)
# rk = np.array(rk)
rm = np.mean(rm, axis=1)
# rc = np.mean(rc, axis=1)
# rk = np.mean(rk, axis=1)      # average along the time series, so get average realized mean, var, skew, kurt w.r.t. each path
print(rm)
# print(rc)
# print(rk)

# computed_skewness = rc[:,2] / rc[:,1]**(3/2)
# print(f'Computed Skewness is: {computed_skewness}, vs True Skewness is: {skewness}')
end_time = datetime.now()
print(f'Computation Duration is: {end_time - start_time}')