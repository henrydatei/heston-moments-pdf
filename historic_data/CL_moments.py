# Implementation of moments from Choe & Lee (2014)
# r is array of log returns

import pandas as pd
import yfinance as yf
import numpy as np

def quadratic_variation(r: pd.Series) -> float:
    return r.diff().dropna().pow(2).sum()

def quadratic_covariation(r1: pd.Series, r2: pd.Series) -> float:
    return (r1.diff().dropna() * r2.diff().dropna()).sum()

def realized_variance(r: pd.Series) -> float:
    '''Andersen et al. (2003), Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P. (2001).'''
    return quadratic_variation(r)

def realized_third_moment(r: pd.Series) -> float:
    return 1.5 * quadratic_covariation(r**2, r)

def realized_fourth_moment(r: pd.Series) -> float:
    return 1.5 * quadratic_variation(r**2)

def realized_skewness(r: pd.Series) -> float:
    '''uses noncentral moments'''
    return realized_third_moment(r) / realized_variance(r)**1.5

def realized_kurtosis(r: pd.Series) -> float:
    '''uses noncentral moments'''
    return realized_fourth_moment(r) / realized_variance(r)**2


data = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

print(realized_variance(data['log_return']))
print(realized_skewness(data['log_return']))
print(realized_kurtosis(data['log_return']))
