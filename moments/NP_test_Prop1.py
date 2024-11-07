# I want to test, whether Prop 1 from Neuberger & Payne 2021 holds. They say that var[D(T)] = T * var[d], with daily prices changes d and monthly prices changes D with T days in each month. As we see, with more months this holds, with less months it doesn't hold.

# also test statements about skewness and kurtosis

import pandas as pd
import numpy as np
import scipy
import scipy.stats

number_of_months = 10000
T = 20 # number of days per month

# Generate a month's worth of prices with random fluctuations
np.random.seed(0)  # For reproducibility
initial_price = 100  # Starting price

# Generate prices
total_days = number_of_months * T
# daily_returns = np.random.normal(0, 1, total_days)  # Daily returns with mean 0 and std 1
daily_returns = np.random.standard_t(10, total_days)

# Generate prices based on daily returns
prices = [initial_price]
for daily_return in daily_returns:
    prices.append(prices[-1] + daily_return)

# Calculate daily differences (price changes)
daily_changes = np.diff(prices)

# shift daily changes by 1 day and add a zero at the beginning
daily_changes = np.insert(daily_changes, 0, 0)
daily_changes = np.roll(daily_changes, 1)[1:]

# Monthly total changes (D(T)) for each month
# monthly_changes = [prices[i*T+T] - prices[i*T] for i in range(number_of_months)]
# monthly_changes = np.array(monthly_changes)

# monthly changes as price differences from T days ago
monthly_changes = np.array([prices[i] - prices[i-T] for i in range(T, total_days)])

def y_1_star_t(prices, t, T):
    if t <= T:
        # we yesterdays return with the return of the last month (last T days)
        return 0
    sum = 0
    for u in range(1,T+1):
        sum += prices[t-1] - prices[t-u]
    return sum / T

# A different implementation of y_1_star_t, which is equivalent to the one above
def D(prices, subscript, argument):
    return prices[subscript] - prices[subscript - argument]

# A different implementation of y_1_star_t, which is equivalent to the one above
def y_1_star_t_2(prices, t, T):
    if t <= T:
        # we yesterdays return with the return of the last month (last T days)
        return 0
    sum = 0
    for u in range(1,T):
        sum += D(prices, t-1, u)
    return sum / T

def y_1_star(prices, T):
    array = []
    for t in range(len(prices)):
        array.append(y_1_star_t(prices, t, T))
    return array[1:]

# A different implementation of y_1_star, which is equivalent to the one above
def y_1_star_2(prices, T):
    array = []
    for t in range(len(prices)):
        array.append(y_1_star_t_2(prices, t, T))
    return array[1:]

def y_2_star_t(prices, t, T):
    if t <= T:
        # we yesterdays return with the return of the last month (last T days)
        return 0
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
# y_1star_2 = y_1_star_2(prices, T)
y_2star = y_2_star(prices, T)

# print(prices_df)

# print(monthly_changes)

print("Test Prop 1")
print(var_d, var_D_T, T * var_d)
print(skew_d, skew_D_T, (skew_d + 3*(np.cov(y_1star, daily_changes**2)[1,0])/(var_d**1.5))/np.sqrt(T))
print(kurt_d, kurt_D_T, (kurt_d + 4*(np.cov(y_1star, daily_changes**3)[1,0])/(var_d**2) + 6*(np.cov(y_2star, daily_changes**2)[1,0])/(var_d**2))/T)

# print(y_1star == y_1star_2) # they are equal

# we find in the appendix of the paper 
# E(D^2) = T * E(d^2) and
# E(D^3) = T * (E(d^3) + 3*cov(y_1_star, d^2)) and 
# E(D^4) - 3E(D^2)^2 = T * (E(d^4) - 3E(d^2)^2 + 4*cov(y_1_star, d^3) + 6*cov(y_2_star, d^2))

print("Final equations from proof from Prop 1 (A5)")
print(np.mean(monthly_changes**2), T * np.mean(daily_changes**2))
print(np.mean(monthly_changes**3), T * (np.mean(daily_changes**3) + 3*np.cov(y_1star, daily_changes**2)[1,0]))
print(np.mean(monthly_changes**4) - 3*np.mean(monthly_changes**2)**2, T * (np.mean(daily_changes**4) - 3*np.mean(daily_changes**2)**2 + 4*np.cov(y_1star, daily_changes**3)[1,0] + 6*np.cov(y_2star, daily_changes**2)[1,0]))

print("Test expectations from y_1_star and y_2_star (A4)")
print(np.mean(y_1star), 0)
print(np.mean(y_2star), 0.5*(T-1)*np.mean(daily_changes**2))

print("Test unconditional expectations (A3)")
print(np.mean(monthly_changes**2), T * np.mean(daily_changes**2))
print(np.mean(monthly_changes**3), T * np.mean(daily_changes**3) + 3*T*np.mean(daily_changes**2 * y_1star))
print(np.mean(monthly_changes**4), T*np.mean(daily_changes**4) + 4*T*np.mean(daily_changes**3 * y_1star) + 6*T*np.mean(daily_changes**2 * y_2star))

print("Test decomposition of monthly price change (A1)")
print("difference in decomposition of D_t^2")
decomposed_D2_t = []
for t in range(T, total_days):
    sum1 = 0
    sum2 = 0
    for u in range(0, T):
        sum1 += daily_changes[t-u]**2
        sum2 += (prices[t-u-1] - prices[t-T]) * daily_changes[t-u]
    decomposed_D2_t.append(sum1 + 2*sum2)
# print(decomposed_D2_t)
# print(monthly_changes**2)
print(decomposed_D2_t - monthly_changes**2)
print(all(decomposed_D2_t == monthly_changes**2))

print("difference in decomposition of D_t^3")
decomposed_D3_t = []
for t in range(T, total_days):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for u in range(0, T):
        sum1 += daily_changes[t-u]**3
        sum2 += (prices[t-u-1] - prices[t-T]) * daily_changes[t-u]**2
        sum3 += (prices[t-u-1] - prices[t-T])**2 * daily_changes[t-u]
    decomposed_D3_t.append(sum1 + 3*sum2 + 3*sum3)
# print(decomposed_D3_t)
# print(monthly_changes**3)
print(decomposed_D3_t - monthly_changes**3)
print(all(decomposed_D3_t == monthly_changes**3))

print("difference in decomposition of D_t^4")
decomposed_D4_t = []
for t in range(T, total_days):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for u in range(0, T):
        sum1 += daily_changes[t-u]**4
        sum2 += (prices[t-u-1] - prices[t-T]) * daily_changes[t-u]**3
        sum3 += (prices[t-u-1] - prices[t-T])**2 * daily_changes[t-u]**2
        sum4 += (prices[t-u-1] - prices[t-T])**3 * daily_changes[t-u]
    decomposed_D4_t.append(sum1 + 4*sum2 + 6*sum3 + 4*sum4)
# print(decomposed_D4_t)
# print(monthly_changes**4)
print(decomposed_D4_t - monthly_changes**4)
print(all(decomposed_D4_t == monthly_changes**4))