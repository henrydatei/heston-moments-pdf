import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments
from moments.ACJV_moments import realized_daily_variance, realized_daily_skewness, realized_daily_kurtosis, realized_daily_mean, realized_daily_skewness_2, realized_daily_kurtosis_2
from moments.CL_moments import realized_variance, realized_skewness, realized_kurtosis, realized_mean, realized_third_moment, realized_fourth_moment

np.random.seed(0)

time_points = 3 * 12 * 22 * 79 # 3 years with 12 months with 22 trading days with 79 prices (5 minute intervals from 9:30 am to 4pm and inital price
T = 3
S0 = 100
paths = 1

QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, N=time_points, n_paths=paths) # the first point is lnS0
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

print(pd.DataFrame(rm).T)

# plt.hist(df_logreturn['Path_1'], bins=100)
# plt.show()

# print(skew(df_logreturn['Path_1']), kurtosis(df_logreturn['Path_1']))


# Henrys implementation of ACJV moments
# realized_moments = np.zeros((8, len(df_logreturn.columns)))

# # Iterate over each column and compute the required moments
# for i, column in enumerate(df_logreturn.columns):
#     log_return_series = df_logreturn[column]
    
#     # Compute rolling sums and means
#     rolling_sum_mean = realized_daily_mean(log_return_series).rolling(22).sum().dropna()
#     rolling_sum_variance = realized_daily_variance(log_return_series).rolling(22).sum().dropna()
#     rolling_sum_skewness = realized_daily_skewness(log_return_series).rolling(22).sum().dropna()
#     rolling_sum_kurtosis = realized_daily_kurtosis(log_return_series).rolling(22).sum().dropna()
#     rolling_sum_skewness_2 = realized_daily_skewness_2(log_return_series).rolling(22).sum().dropna() # without multiplication by sqrt(prices_per_day)
#     rolling_sum_kurtosis_2 = realized_daily_kurtosis_2(log_return_series).rolling(22).sum().dropna() # without multiplication by prices_per_day

#     # Calculate realized moments and store them in the array
#     realized_moments[0, i] = rolling_sum_mean.mean()
#     realized_moments[1, i] = rolling_sum_variance.mean()
#     realized_moments[2, i] = rolling_sum_skewness.mean()
#     realized_moments[3, i] = rolling_sum_kurtosis.mean()
#     realized_moments[4, i] = rolling_sum_skewness_2.mean()
#     realized_moments[5, i] = rolling_sum_kurtosis_2.mean()
    
#     # Haozhe's implementation
#     realized_moments[6, i] = (log_return_series**3).resample('D').sum(min_count=1).dropna().rolling(22).sum().dropna().mean() / (rolling_sum_variance.mean()**(3/2))
#     realized_moments[7, i] = (log_return_series**4).resample('D').sum(min_count=1).dropna().rolling(22).sum().dropna().mean() / (rolling_sum_variance.mean()**2)

# # Print the resulting array
# print(pd.DataFrame(realized_moments))

# Henrys implementation of CL moments
realized_moments = np.zeros((6, len(df_logreturn.columns)))

# Iterate over each column and compute the required moments
for i, column in enumerate(df_logreturn.columns):
    log_return_series = df_logreturn[column]
    daily_variance = log_return_series.resample('D').apply(realized_variance)
    daily_skewness = log_return_series.resample('D').apply(realized_skewness)
    daily_kurtosis = log_return_series.resample('D').apply(realized_kurtosis)
    # remove weekends
    daily_variance = daily_variance[daily_variance.index.weekday < 5]
    daily_skewness = daily_skewness[daily_skewness.index.weekday < 5]
    daily_kurtosis = daily_kurtosis[daily_kurtosis.index.weekday < 5]
    # Calculate realized moments and store them in the array
    realized_moments[0, i] = realized_mean(log_return_series)
    realized_moments[1, i] = daily_variance.rolling(22).sum().dropna().mean()
    realized_moments[2, i] = daily_skewness.rolling(22).sum().dropna().mean()
    realized_moments[3, i] = daily_kurtosis.rolling(22).sum().dropna().mean()
    # Haozhe's implementation
    daily_third = log_return_series.resample('D').apply(realized_third_moment)
    daily_fourth = log_return_series.resample('D').apply(realized_fourth_moment)
    daily_third = daily_third[daily_third.index.weekday < 5]
    daily_fourth = daily_fourth[daily_fourth.index.weekday < 5]
    realized_moments[4, i] = daily_third.rolling(22).sum().dropna().mean() / (daily_variance.rolling(22).sum().dropna().mean()**(3/2))
    realized_moments[5, i] = daily_fourth.rolling(22).sum().dropna().mean() / (daily_variance.rolling(22).sum().dropna().mean().mean()**2)

# Print the resulting array
print(pd.DataFrame(realized_moments))