import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the data for the S&P 500 index
data = yf.download('^GSPC', start='1900-01-01', end='2023-12-31')

# returns
data['return'] = data['Close'].pct_change()

# delete the first row
data = data.iloc[1:]

data['Year'] = data.index.year
# Group by 'Year' and get the first and last 'Close' price of each year
yearly_close = data.groupby('Year')['Close'].agg(['first', 'last'])
yearly_close['return'] = (yearly_close['last'] - yearly_close['first']) / yearly_close['first']

print(data)

# first 4 moments of returns, short term
mean_short_term_return = data['return'].mean()
var_short_term_return = data['return'].var()
skew_short_term_return = data['return'].skew()
kurt_short_term_return = data['return'].kurt()

# first 4 moments of returns, long term
mean_long_term_return = yearly_close['return'].mean()
var_long_term_return = yearly_close['return'].var()
skew_long_term_return = yearly_close['return'].skew()
kurt_long_term_return = yearly_close['return'].kurt()

print('Moments for Returns:')
print(f'Short term mean: {mean_short_term_return:.4f}, long term mean: {mean_long_term_return:.4f}')
print(f'Short term variance: {var_short_term_return:.4f}, long term variance: {var_long_term_return:.4f}')
print(f'Short term skewness: {skew_short_term_return:.4f}, long term skewness: {skew_long_term_return:.4f}')
print(f'Short term kurtosis: {kurt_short_term_return:.4f}, long term kurtosis: {kurt_long_term_return:.4f}')

# How moments change over time
means = []
variances = []
skewnesses = []
kurtoses = []

# Calculate statistics for rolling windows from 1 to the length of the DataFrame
for number_datapoints in tqdm(range(1, len(data) + 1)):
    reduced_data = data['return'].iloc[:number_datapoints]
    
    # Calculate statistics
    means.append(reduced_data.mean())
    variances.append(reduced_data.var())
    skewnesses.append(reduced_data.skew())
    kurtoses.append(reduced_data.kurt())

# Mean plot
plt.subplot(4, 1, 1)
plt.plot(range(1, len(data) + 1), means, label='Mean')
plt.title('Rolling Statistics Over Increasing Windows')
plt.ylabel('Mean')
plt.grid(True)

# Variance plot
plt.subplot(4, 1, 2)
plt.plot(range(1, len(data) + 1), variances, label='Variance', color='orange')
plt.ylabel('Variance')
plt.grid(True)

# Skewness plot
plt.subplot(4, 1, 3)
plt.plot(range(1, len(data) + 1), skewnesses, label='Skewness', color='green')
plt.ylabel('Skewness')
plt.grid(True)

# Kurtosis plot
plt.subplot(4, 1, 4)
plt.plot(range(1, len(data) + 1), kurtoses, label='Kurtosis', color='red')
plt.ylabel('Kurtosis')
plt.xlabel('Window Size (Days)')
plt.grid(True)

plt.tight_layout()
plt.show()