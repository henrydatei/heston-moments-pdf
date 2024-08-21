# Implementation of moments from Amaya et al (2015)
# r is array of log returns

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

def realized_daily_variance(r: pd.Series, prices_per_day: int) -> pd.Series:
    '''square rows of r, sum over non overlapping blocks of length prices_per_day'''
    r_squared = r**2
    return r_squared.groupby(np.arange(len(r_squared))//prices_per_day).sum()

def realized_daily_skewness(r: pd.Series, prices_per_day: int) -> pd.Series:
    '''cube rows of r, sum over non overlapping blocks of length prices_per_day, divide by realized_daily_variance(r, prices_per_day)^(3/2), multiply by sqrt(prices_per_day)'''
    r_cubeed = r**3
    return r_cubeed.groupby(np.arange(len(r_cubeed))//prices_per_day).sum()/realized_daily_variance(r, prices_per_day)**(3/2) * np.sqrt(prices_per_day)

def realized_daily_kurtosis(r: pd.Series, prices_per_day: int) -> pd.Series:
    '''rows of r^4, sum over non overlapping blocks of length prices_per_day, divide by realized_daily_variance(r, prices_per_day)^2, multiply by prices_per_day'''
    r4 = r**4
    return r4.groupby(np.arange(len(r4))//prices_per_day).sum()/realized_daily_variance(r, prices_per_day)**2 * prices_per_day

def realized_weekly_variance(r: pd.Series, prices_per_day: int, days_per_week: int, annualized: bool = True) -> pd.Series:
    '''rolling mean of realized_daily_variance(r, prices_per_day) * 252 (if annualized)'''
    rolling_mean_variance = realized_daily_variance(r, prices_per_day).rolling(days_per_week).mean()
    if annualized:
        return rolling_mean_variance * 252
    else:
        return rolling_mean_variance
    
def realized_weekly_skewness(r: pd.Series, prices_per_day: int, days_per_week: int) -> pd.Series:
    '''rolling mean of realized_daily_skewness(r, prices_per_day)'''
    return realized_daily_skewness(r, prices_per_day).rolling(days_per_week).mean()

def realized_weekly_kurtosis(r: pd.Series, prices_per_day: int, days_per_week: int) -> pd.Series:
    '''rolling mean of realized_daily_kurtosis(r, prices_per_day)'''
    return realized_daily_kurtosis(r, prices_per_day).rolling(days_per_week).mean()
    
data = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

data = data.iloc[1:]

print(realized_weekly_variance(data['log_return'], 25, 1))
print(realized_weekly_skewness(data['log_return'], 25, 1))
print(realized_weekly_kurtosis(data['log_return'], 25, 1))
