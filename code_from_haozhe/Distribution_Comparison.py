# Compare the distribution from expansion method against the true one

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DensityRecoveryFFT import RecoverDensity
from CharacteristicFunction import ChF_Bates_Gatheral
from HestonDensity_FFT import HestonChfDensity_FFT, HestonChfDensity_FFT_Gatheral
from GramCharlier_expansion import Expansion_GramCharlier
# from pytorch_lightning import seed_everything
from SimHestonQE import Heston_QE
from TimeTruncation import align_to_business_days
import seaborn as sns
from scipy import stats

# plot the Heston monthly log-return distribution
f_x = HestonChfDensity_FFT_Gatheral(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)
#print(f_x)
# get a density using the true cumulant
true_cumulant = np.array([-0.00791667, 0.01601168, -0.00056375, 0.00088632])
f_true_x = Expansion_GramCharlier(true_cumulant)

# one sample cumulant from the realized cumulant method
cumulant = np.array([-0.01401158, 0.01647651, -0.0006112, 0.00093973])
f_hat_x = Expansion_GramCharlier(cumulant)
#print(f_hat_x)
# get an empirical distribution
print('*****************************************************Simulation*****************************************************************')
# seed_everything(seed=33)
time_points = 60 * 22 * 12    
burnin = 10 * 22 * 12
T = 60         
S0 = 100
paths = 1
QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, n=time_points, M=paths)    # the first point is lnS0
QE_process = np.diff(QE_process)                    # to get the difference, i.e., log-returns
QE_process_cut = QE_process[:, burnin:]         # set the first 2 years and 10 months as burnin
QE_process_transform = QE_process_cut.T         # need to transform it to set the index

# Create a DatetimeIndex for every 5 minutes during trading hours
start_date = '2014-07-01'
end_date = '2076-07-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
#time_range = pd.date_range(start='09:30', end='16:05', freq='5T')[:-1]  # Trading hours making it actually end at 16:00

# Create a full DatetimeIndex for all trading days and intraday times
datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()}') for day in date_range])

# Ensure the datetime_index length matches the data length
time_index = time_points - burnin
datetime_index = datetime_index[:time_index]

# Create the DataFrame
df_logreturn = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process_cut.shape[0])])

# make sure the first point is the first business day and the last point is the last business day of a certain month
cut_df_logreturn = align_to_business_days(df_logreturn)
#print(cut_df_logreturn)

# get the monthly return
monthly_logreturn = cut_df_logreturn.resample('M').sum()
data = monthly_logreturn.to_numpy()

x = np.linspace(-2.0, 2.0, 1000)
q = 1000       # quantile
# plot the density comparison
plt.figure(figsize=(12, 8))
plt.grid()
plt.xlabel("x")
plt.ylabel("$f(x)$")
#sns.histplot(data, label='Empirical', bins=30, kde=True, stat='density', palette=['gold'])
sns.kdeplot(data, label='Empirical', fill=True, palette=['gold'])
plt.plot(x[:q], f_x[:q], label=r'$f(x)$', linestyle='-.', color='blue', linewidth=2)
plt.plot(x[:q], f_true_x[:q], label=r'$\tilde{f}(x) using\ true\ cumulants$', linestyle='-', color='red', linewidth=2)
plt.plot(x[:q], f_hat_x[:q], label=r'$\hat{f}(x) using\ realized\ cumulants$', linestyle='--', color='green', linewidth=2)

# Adding a legend in the upper left corner
plt.legend(loc='upper left')

# plt.savefig('Comparison_Density.pdf') 
plt.show()


# conduct a Kolmogorov-Smirnov test

# Approximate CDF by taking the cumulative sum of the PDF values
# Scale by the step size (dx) between points
dx = x[1] - x[0]
true_cdf = np.cumsum(f_x) * dx
approxi_cdf = np.cumsum(f_hat_x) * dx

# Normalize the CDF (since the sum of the PDF over all space should equal 1)
true_cdf, approxi_cdf = true_cdf / true_cdf[-1], approxi_cdf / approxi_cdf[-1]

# # check cdf plot
# plt.plot(x, true_cdf, label="True CDF")
# plt.plot(x, approxi_cdf, label="NP CDF", linestyle='dashed')
# plt.legend()
# plt.show()

# Generate uniform random numbers
n_samples = 1000  # Number of samples you want to generate
uniform_randoms = np.random.uniform(0, 1, n_samples)  # Uniform random numbers

# Generate samples from the empirical CDF
generated_true_samples = np.zeros(n_samples)
generated_approxi_samples = np.zeros(n_samples)
for i in range(n_samples):
    # Find the corresponding value in the empirical CDF
    generated_true_samples[i] = x[np.searchsorted(true_cdf, uniform_randoms[i])]
    generated_approxi_samples[i] = x[np.searchsorted(approxi_cdf, uniform_randoms[i])]

# Perform the K-S test 
ks_statistic, p_value = stats.kstest(generated_approxi_samples, generated_true_samples)
#ks_statistic, p_value = stats.ks_2samp(approxi_cdf, true_cdf)

print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")


# Perform the Cramér-von Mises test

res = stats.cramervonmises_2samp(generated_approxi_samples, generated_true_samples)
statistic, p_value = res.statistic, res.pvalue

# Output the results
print("Cramér-von Mises Statistic:", statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The realized distribution and the true distribution are significantly different.")
else:
    print("Fail to reject the null hypothesis: The realized distribution and the true distribution are not significantly different.")


# # Perform the Chi-Square Goodness-of-Fit Test

# # Calculate probabilities for each interval using the CDF
# # The probabilities are given by F(x2) - F(x1) for each interval [x1, x2)
# true_probabilities = true_cdf[1:] - true_cdf[:-1]
# approxi_probabilities = approxi_cdf[1:] - approxi_cdf[:-1]

# total_samples = 1000

# # Calculate expected frequencies
# true_frequencies = true_probabilities * total_samples
# approxi_frequencies = approxi_probabilities * total_samples

# true_frequencies = true_frequencies / true_frequencies.sum() * approxi_frequencies.sum()   # ensure  the sum observed frequencies exactly matches the sum of expected frequencies

# chi2_statistic, p_value = stats.chisquare(f_obs=approxi_frequencies, f_exp=true_frequencies)

# # Output the results
# print("Chi-Squared Statistic:", chi2_statistic)
# print("P-value:", p_value)

# # Interpret the results
# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis: The observed frequencies do not match the expected frequencies.")
# else:
#     print("Fail to reject the null hypothesis: The observed frequencies match the expected frequencies.")


# make a Q-Q plot

quantiles1 = np.sort(generated_true_samples)
quantiles2 = np.sort(generated_approxi_samples)

plt.figure(figsize=(12,8))
plt.plot(quantiles1, quantiles2, 'o', markersize=3, label='Q-Q Plot')
plt.plot([min(quantiles1.min(), quantiles2.min()), max(quantiles1.max(), quantiles2.max())], 
         [min(quantiles1.min(), quantiles2.min()), max(quantiles1.max(), quantiles2.max())], 
         color='red', lw=2, label='y=x (Perfect Fit)')

plt.xlabel('Quantiles of True Distribution')
plt.ylabel('Quantiles of Realized Distribution')
plt.title('Q-Q Plot: Realized Distribution vs True Distribution')
plt.legend()
plt.grid(True)
# plt.savefig('Comparison_QQ.pdf') 
plt.show()