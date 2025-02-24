import numpy as np
import pandas as pd
from RealizedSkewness_NP import rCumulants, rSkewness
from RealizedSkewness_NP_MonthlyOverlap import rCumulants_MonthlyOverlap, rSkewness_MonthlyOverlap

def NP_Estimator_MonthlyOverlap(QE_process, time_points, burnin, log_return=True):   # here the input is the simulated paths of log returns if Ture, otherwise: log prices
    if log_return:
        QE_process = QE_process
    else:
        QE_process = np.diff(QE_process)                # to get the difference, i.e., log-returns

    QE_process_transform = QE_process.T         # need to transform it to set the index

    # Create a DatetimeIndex for every 5 minutes during trading hours
    start_date = '2014-07-01'
    end_date = '2026-07-01'
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    time_range = pd.date_range(start='09:30', end='16:05', freq='5min')[:-1]  # Trading hours making it actually end at 16:00

    # Create a full DatetimeIndex for all trading days and intraday times
    datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

    # Ensure the datetime_index length matches the data length
    time_index = time_points - burnin
    datetime_index = datetime_index[:time_index]

    # Create the DataFrame
    df_logreturn = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process.shape[0])])

    # make sure the first point is the first business day and the last point is the last business day of a certain month
    cut_df_logreturn = align_to_business_days(df_logreturn)

    # Define constants for the realized moments
    RM_ACJV = 1  # Amaya et al, 2015
    RM_CL = 2    # Choe and Lee, 2014
    RM_NP = 3    # Neuberger and Payne, 2018
    RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
    RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

    # Compute realized moments using NP return method
    technique = RM_NP
    rc, rk = ([] for i in range(2))
    for column in df_logreturn.columns:
        rc.append(rCumulants_MonthlyOverlap(cut_df_logreturn[column], method=technique, months_overlap=2).to_numpy())
        #rk.append(rSkewness_MonthlyOverlap(cut_df_logreturn[column], method=technique, months_overlap=6).to_numpy())
    rc = np.squeeze(np.array(rc))           # no need to squeeze rk because it has one element which has been automatically squeezed by np
    return rc


def NP_Estimator_single(QE_process, log_return=True):   # here the input is the simulated paths of log returns if Ture, otherwise: log prices
    if log_return:
        QE_process = QE_process
    else:
        QE_process = np.diff(QE_process)                # to get the difference, i.e., log-returns

    QE_process_transform = QE_process.T         # need to transform it to set the index

    # Create a DatetimeIndex for every 5 minutes during trading hours
    start_date = '2014-07-01'
    end_date = '2026-07-01'
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    time_range = pd.date_range(start='09:30', end='16:05', freq='5min')[:-1]  # Trading hours making it actually end at 16:00

    # Create a full DatetimeIndex for all trading days and intraday times
    datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

    # Ensure the datetime_index length matches the data length
    time_index = 2 * 22 * 79
    datetime_index = datetime_index[:time_index]

    # Create the DataFrame
    df_logreturn = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process.shape[0])])

    # Define constants for the realized moments
    RM_ACJV = 1  # Amaya et al, 2015
    RM_CL = 2    # Choe and Lee, 2014
    RM_NP = 3    # Neuberger and Payne, 2018
    RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
    RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

    # Compute realized moments using NP return method
    technique = RM_NP
    rc, rk = ([] for i in range(2))
    for column in df_logreturn.columns:
        rc.append(rCumulants(df_logreturn[column], method=technique, days_aggregate=22).to_numpy())
        #rk.append(rSkewness(df_logreturn[column], method=technique, days_aggregate=22).to_numpy())
    rc = np.squeeze(np.array(rc))           # no need to squeeze rk because it has one element which has been automatically squeezed by np
    return rc


def align_to_business_days(df):
    # Check if the DataFrame has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Align the first point
    while True:
        # Get the first date in the index
        first_date = df.index[0]
        
        # Get the first business day of that month
        first_business_day = pd.date_range(first_date, periods=1, freq='BMS')[0]
        
        if first_date == first_business_day:
            break
        else:
            # Drop the first row if the first date is not the first business day
            df = df.iloc[1:]

    # Align the last point
    while True:
        # Get the last date in the index
        last_date = df.index[-1]
        
        # Get the last business day of that month
        last_business_day = pd.date_range(last_date, periods=1, freq='BME')[0]   # before "BM" is deprecated
        
        if last_date == last_business_day:
            break
        else:
            # Drop the last row if the last date is not the last business day
            df = df.iloc[:-1]
    
    return df