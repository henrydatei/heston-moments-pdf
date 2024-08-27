# Implementation of moments from Amaya et al (2015)
# r is array of log returns

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

def realized_daily_mean(r: pd.Series) -> pd.Series:
    '''sum over a day (r needs to be a pd.Series with a DatetimeIndex)'''
    return r.resample('D').sum(min_count=1).dropna()

def realized_daily_variance(r: pd.Series) -> pd.Series:
    '''square rows of r, sum over a day (r needs to be a pd.Series with a DatetimeIndex)'''
    return (r**2).resample('D').sum(min_count=1).dropna()

def realized_daily_skewness(r: pd.Series) -> pd.Series:
    '''cube rows of r, sum over a day (r needs to be a pd.Series with a DatetimeIndex), divide by realized_daily_variance(r)^(3/2), multiply by sqrt(prices_per_day)'''
    prices_per_day = r.resample('D').count().mode()[0] # if we have days with less prices, we just assume that these days have the same amount of prices as the most common day
    return (r**3).resample('D').sum(min_count=1).dropna()/(realized_daily_variance(r)**(3/2)) * np.sqrt(prices_per_day)

def realized_daily_kurtosis(r: pd.Series) -> pd.Series:
    '''rows of r^4, sum over a day (r needs to be a pd.Series with a DatetimeIndex), divide by realized_daily_variance(r)^2, multiply by prices_per_day'''
    prices_per_day = r.resample('D').count().mode()[0] # if we have days with less prices, we just assume that these days have the same amount of prices as the most common day
    return (r**4).resample('D').sum(min_count=1).dropna()/(realized_daily_variance(r)**2) * prices_per_day

def realized_daily_skewness_2(r: pd.Series) -> pd.Series:
    '''cube rows of r, sum over a day (r needs to be a pd.Series with a DatetimeIndex), divide by realized_daily_variance(r)^(3/2)'''
    return (r**3).resample('D').sum(min_count=1).dropna()/(realized_daily_variance(r)**(3/2))

def realized_daily_kurtosis_2(r: pd.Series) -> pd.Series:
    '''rows of r^4, sum over a day (r needs to be a pd.Series with a DatetimeIndex), divide by realized_daily_variance(r)^2'''
    return (r**4).resample('D').sum(min_count=1).dropna()/(realized_daily_variance(r)**2)

def realized_weekly_variance(r: pd.Series, days_per_week: int, annualized: bool = True) -> pd.Series:
    '''rolling mean of realized_daily_variance(r) * 252 (if annualized)'''
    rolling_mean_variance = realized_daily_variance(r).rolling(days_per_week).mean()
    if annualized:
        return rolling_mean_variance * 252
    else:
        return rolling_mean_variance
    
def realized_weekly_skewness(r: pd.Series, days_per_week: int) -> pd.Series:
    '''rolling mean of realized_daily_skewness(r)'''
    return realized_daily_skewness(r).rolling(days_per_week).mean()

def realized_weekly_kurtosis(r: pd.Series, days_per_week: int) -> pd.Series:
    '''rolling mean of realized_daily_kurtosis(r)'''
    return realized_daily_kurtosis(r).rolling(days_per_week).mean()
    
# data = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')

# data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# data = data.iloc[1:]

# print(realized_weekly_variance(data['log_return'], 25, 1))
# print(realized_weekly_skewness(data['log_return'], 25, 1))
# print(realized_weekly_kurtosis(data['log_return'], 25, 1))
