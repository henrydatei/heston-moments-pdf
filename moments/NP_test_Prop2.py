# I want to test, whether Prop 2 from Neuberger & Payne 2021 holds.

import pandas as pd
import numpy as np

number_of_months = 1000
T = 20 # number of days per month

# Generate a month's worth of prices with random fluctuations
np.random.seed(0)  # For reproducibility
initial_price = 1000  # Starting price

# Generate prices
total_days = number_of_months * T
daily_returns = np.random.normal(0, 1, total_days)  # Daily returns with mean 0 and std 1

# Generate prices based on daily returns
prices = [initial_price]
for daily_return in daily_returns:
    prices.append(prices[-1] + daily_return)

# Calculate daily differences (price changes)
daily_changes = np.diff(prices)

# Monthly total changes (D(T)) for each month
monthly_changes = [prices[i*T+T] - prices[i*T] for i in range(number_of_months)]

# daily log returns
r = [np.log(prices[i+1] / prices[i]) for i in range(total_days)]
r = pd.Series(r)

# monthly log returns R(T)
R = [np.log(prices[i*T+T] / prices[i*T]) for i in range(number_of_months)]
R = pd.Series(R)

def x_1(r: pd.Series|float) -> pd.Series|float:
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

def var_L(r: pd.Series) -> float:
    return np.mean(x_2L(r))

def var_E(r: pd.Series) -> float:
    return np.mean(x_2E(r))

def skew(r: pd.Series) -> float:
    return np.mean(x_3(r)) / var_L(r)**(3/2)

def kurt(r: pd.Series) -> float:
    return np.mean(x_4(r)) / var_L(r)**2

def D_for_returns(log_returns: pd.Series, subscript: int, argument: int) -> float:
    '''
    In the paper they use R, but I used R already for monthly returns. I have no idea what R is supposed to be, I assumed it is the same as D in Prop 1.
    '''
    # return log_returns[subscript] - log_returns[subscript - argument]
    
    # make prices from log returns
    p = prices[1:]
    return np.log(p[subscript] / p[subscript - argument])

def y_1_t(r: pd.Series, t: int, T: int):
    if t <= T:
        # y_1_t compares yesterdays return with the return of the last month (last T days)
        return 0
    sum = 0
    for u in range(1,T):
        sum += x_1(D_for_returns(r, t-1, u))
    return sum / T

def y_1(r: pd.Series, T: int):
    array = []
    for t in range(len(r)):
        array.append(y_1_t(r, t, T))
    return array

def y_2L_t(r: pd.Series, t: int, T: int):
    if t <= T:
        # y_1_t compares yesterdays return with the return of the last month (last T days)
        return 0
    sum = 0
    for u in range(1,T+1):
        sum += x_2L(D_for_returns(r, t-1, u))
    return sum / T

def y_2L(r: pd.Series, T: int):
    array = []
    for t in range(len(r)):
        array.append(y_2L_t(r, t, T))
    return array

# Display results
prices_df = pd.DataFrame({
    "Day": range(1, total_days + 1),
    "Price": prices[1:],
    "log return": r
})

# print(prices_df)

# print(R)

# print(len(y_1(r, T)), len(x_2E(r)))

print(var_L(r), var_L(R), T * var_L(r))

print(skew(r), skew(R), (skew(r) + 3*(np.cov(y_1(r, T),x_2E(r))[1,0])/(var_L(r)**1.5))/np.sqrt(T))

print(kurt(r), kurt(R), (kurt(r) + 4*(np.cov(y_1(r, T),x_3(r))[1,0])/(var_L(r)**2) + 3*(np.cov(y_2L(r, T),x_2L(r))[1,0])/(var_L(r)**2))/T)