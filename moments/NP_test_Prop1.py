# I want to test, whether Prop 1 from Neuberger & Payne 2021 holds. They say that var[D(T)] = T * var[d], with daily prices changes d and monthly prices changes D with T days in each month. As we see, with more months this holds, with less months it doesn't hold.

# also test statements about skewness and kurtosis

import pandas as pd
import numpy as np
import scipy
import scipy.stats

number_of_months = 2
T = 20 # number of days per month

# Generate a month's worth of prices with random fluctuations
np.random.seed(0)  # For reproducibility
initial_price = 100  # Starting price

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

def y_1_star_t(prices, t, T):
    sum = 0
    for u in range(1,T+1):
        sum += prices[t-1] - prices[t-u]
    return sum / T

def D(prices, subscript, argument):
    return prices[subscript] - prices[subscript - argument]

def y_1_star_t_2(prices, t, T):
    sum = 0
    for u in range(1,T):
        sum += D(prices, t-1, u)
    return sum / T

def y_1_star(prices, T):
    array = []
    for t in range(len(prices)):
        array.append(y_1_star_t(prices, t, T))
    return array[1:]

def y_1_star_2(prices, T):
    array = []
    for t in range(len(prices)):
        array.append(y_1_star_t_2(prices, t, T))
    return array[1:]

def y_2_star_t(prices, t, T):
    sum = 0
    for u in range(1,T+1):
        sum += (prices[t-1] - prices[t-u])**2
    return sum / T

def y_2_star(prices, T):
    array = []
    for t in range(len(prices)):
        array.append(y_2_star_t(prices, t, T))
    return array[1:]

# daily changes
var_d = np.var(daily_changes, ddof=1)
skew_d = scipy.stats.skew(daily_changes)
kurt_d = scipy.stats.kurtosis(daily_changes)

# Monthly changes
var_D_T = np.var(monthly_changes, ddof=1)
skew_D_T = scipy.stats.skew(monthly_changes)
kurt_D_T = scipy.stats.kurtosis(monthly_changes)

# Display results
prices_df = pd.DataFrame({
    "Day": range(1, total_days + 1),
    "Price": prices[1:],
    "Daily Change (d)": daily_changes
})

y_1star = y_1_star(prices, T)
y_1star_2 = y_1_star_2(prices, T)
y_2star = y_2_star(prices, T)

# print(prices_df)

# print(monthly_changes)

print(var_d, var_D_T, T * var_d)

print(skew_d, skew_D_T, (skew_d + 3*(np.cov(y_1star, daily_changes**2)[1,0])/(var_d**1.5))/np.sqrt(T))

print(kurt_d, kurt_D_T, (kurt_d + 4*(np.cov(y_1star, daily_changes**3)[1,0])/(var_d**2) + 6*(np.cov(y_2star, daily_changes**2)[1,0])/(var_d**2))/T)

print(y_1star)
print(y_1star_2)
print(y_1star == y_1star_2)