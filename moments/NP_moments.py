# Implementation of moments from Neuberger & Payne (2021)
# r is array of log returns

import numpy as np
import pandas as pd
import yfinance as yf

def get_rt(prices: pd.Series, m: float = 0) -> pd.Series:
    '''log daily returns'''
    return np.log(prices / prices.shift(1)) - m

def get_R(prices: pd.Series, T: int, t: int, m: float = 0) -> float:
    return np.log(prices[t] / prices[t - T]) - m * T

def x_1(r: pd.Series) -> pd.Series:
    '''
    x_2L is the same as in Neuberger (2012), Proposition 4, so x_1 must be from Neuberger (2012), Proposition 3
    '''
    return np.exp(r) - 1

def x_2L(r: pd.Series) -> pd.Series:
    return 2 * (np.exp(r) - 1 - r)

def x_2E(r: pd.Series) -> pd.Series:
    return 2 * (r*np.exp(r) - np.exp(r) + 1)

def x_3(r: pd.Series) -> pd.Series:
    return 6 * (r * (np.exp(r) + 1) - 2 * (np.exp(r) - 1))

def x_4(r: pd.Series) -> pd.Series:
    return 12 * (r**2 + 2 * r * (np.exp(r) + 2) - 6 * (np.exp(r) - 1))

def var_L_daily(r: pd.Series) -> float:
    return np.mean(x_2L(r))

def var_E_daily(r: pd.Series) -> float:
    return np.mean(x_2E(r))

def skew_daily(r: pd.Series) -> float:
    return np.mean(x_3(r)) / var_L_daily(r)**(3/2)

def kurt_daily(r: pd.Series) -> float:
    return np.mean(x_4(r)) / var_L_daily(r)**2

def y_1_t(prices: pd.Series, T: int, t: int) -> float:
    sum = 0
    for u in range(1, T):
        sum += x_1(get_R(prices, u, t-1))
    return sum/T

def y_2L_t(prices: pd.Series, T: int, t: int) -> float:
    sum = 0
    for u in range(1, T):
        sum += x_2L(get_R(prices, u, t-1))
    return sum/T

def var_monthly(r: pd.Series, T: int) -> float:
    return T * var_L_daily(r)

def skew_monthly(r: pd.Series, T: int) -> float:
    '''
    skewness of monthly returns. I assume months overlap, so for T=25, we have returns from day 1 to 25, 2 to 26, etc.
    '''
    prices = np.exp(r)
    y_1 = []
    for t in range(T, len(prices)):
        y_1.append(y_1_t(prices, T, t))
    
    cov1 = np.cov(y_1, x_2E(r)[T:])[1,0]
    
    # print(x_2E(r)[T:])
    # print(y_1)
    # print(np.dot(y_1, x_2E(r)[T:]))
    # print(cov1)
    
    return (skew_daily(r) + 3 * cov1 / var_L_daily(r)**(3/2)) / np.sqrt(T)

def kurt_monthly(r: pd.Series, T: int) -> float:
    '''
    excess kurtosis of monthly returns. I assume months overlap, so for T=25, we have returns from day 1 to 25, 2 to 26, etc.
    '''
    prices = np.exp(r)
    y_1 = []
    y_2L = []
    for t in range(T, len(prices)):
        y_1.append(y_1_t(prices, T, t))
        y_2L.append(y_2L_t(prices, T, t))

    cov1 = np.cov(y_1, x_3(r)[T:])[1,0]
    cov2 = np.cov(y_2L, x_2L(r)[T:])[1,0]

    return (kurt_daily(r) + 4 * cov1 / var_L_daily(r)**2 + 6 * cov2 / var_L_daily(r)**2) / T

data = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')
data["log_returns"] = get_rt(data["Close"])

# delete the first row
data = data.iloc[1:]

print(f'variance of log returns daily: {var_L_daily(data["log_returns"])}, monthly: {var_monthly(data["log_returns"], 25)}')
print(f'skewness of log returns daily: {skew_daily(data["log_returns"])}, monthly: {skew_monthly(data["log_returns"], 25)}')
print(f'kurtosis of log returns daily: {kurt_daily(data["log_returns"])}, monthly: {kurt_monthly(data["log_returns"], 25)}')