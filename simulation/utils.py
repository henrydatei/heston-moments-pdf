import pandas as pd
import numpy as np

def process_to_log_returns(process: np.ndarray, start_date: str, end_date: str, burnin_timesteps: int = 2 * 22 * 79 * 12 + 22 * 79 * 10) -> pd.DataFrame:
    QE_process_transform = process[:,1:].T # need to transform it to set the index

    # Create a DatetimeIndex for every 5 minutes during trading hours
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    time_range = pd.date_range(start='09:30', end='16:05', freq='5min')[:-1]  # Trading hours making it actually end at 16:00

    # Create a full DatetimeIndex for all trading days and intraday times
    datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()} {time.time()}') for day in date_range for time in time_range])

    # Ensure the datetime_index length matches the data length
    datetime_index = datetime_index[:QE_process_transform.shape[0]]

    # Create the DataFrame
    df = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process_transform.shape[1])])
    firstday = df.iloc[0]
    df_logreturn = df.diff() # compute the log returns
    df_logreturn.iloc[0] = firstday - np.log(process[0, 0]) # the first log-return is compared to the S0
    
    # set the first 2 years and 10 months as burnin (standard burnin period)
    # we want to estimate unconditional moments, so we need to remove the burnin period because inital parameters are not unconditional
    df_logreturn = df_logreturn.iloc[burnin_timesteps:]

    return df_logreturn