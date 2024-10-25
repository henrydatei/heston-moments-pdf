# I want to test, whether Prop 1 from Neuberger & Payne 2021 holds. They say that var[D(T)] = T * var[d], with daily prices changes d and monthly prices changes D with T days in each month. As we see, with more months this holds, with less months it doesn't hold.

import pandas as pd
import numpy as np

number_of_months = 100
number_of_days_per_month = 20

# Generate a month's worth of prices with random fluctuations
np.random.seed(0)  # For reproducibility
initial_price = 100  # Starting price

# Generate prices
total_days = number_of_months * number_of_days_per_month
daily_returns = np.random.normal(0, 1, total_days)  # Daily returns with mean 0 and std 1

# Generate prices based on daily returns
prices = [initial_price]
for daily_return in daily_returns:
    prices.append(prices[-1] + daily_return)

# Calculate daily differences (price changes)
daily_changes = np.diff(prices)

# Monthly total changes (D(T)) for each month
monthly_changes = [prices[i*number_of_days_per_month+number_of_days_per_month] - prices[i*number_of_days_per_month] for i in range(number_of_months)]

# Variance of daily changes over all days
var_d = np.var(daily_changes, ddof=1)

# Variance of the monthly changes (var[D(T)]) directly calculated
var_D_T = np.var(monthly_changes, ddof=1)

# Display results
prices_df = pd.DataFrame({
    "Day": range(1, total_days + 1),
    "Price": prices[1:],
    "Daily Change (d)": daily_changes
})
# print(prices_df)

# print(monthly_changes)

print(var_d, var_D_T, number_of_days_per_month * var_d)
