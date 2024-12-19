import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.utils import process_to_log_returns_interday
from simulation.SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_CL, RM_NP, RM_ORS, RM_ACJV, RM_NP_return
from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion, edgeworth_expansion, saddlepoint_approximation

np.random.seed(0)

# Define parameters for the simulation
time_points = 3 * 12 * 22 * 79  # 3 years with 12 months with 22 trading days with 79 prices (5 minute intervals from 9:30 am to 4pm and inital price
start_date = '2014-07-01'
end_date = '2026-07-01'
T = 3
S0 = 100
paths = 10
v0 = 0.19
kappa = 3
theta = 0.19
sigma = 0.4
mu = 0 # martingale
rho = -0.7
rolling_window = 22

# Eraker 2004 (for time unit = 1 day)
mu = 0.026 # is not 0, so we need to de-mean the data
theta = 1.933
kappa = 0.019
rho = -0.569
sigma = 0.220

# Eraker 2004 (for time unit = 5 min)
kappa = 0.019 * 1/78
sigma = 0.220 * np.sqrt(1/78)

# Check if Feller condition is satisfied
print(f'Feller condition: {2 * kappa * theta > sigma**2}')

# Simulation
process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
if mu != 0:
    # de-mean the data
    process = process - mu
process_df = process_to_log_returns_interday(process, start_date, end_date)

# Estimate moments
technique = RM_NP_return
mvsek = []
for column in process_df.columns:
    mvsek.append(rMoments_mvsek(process_df[column], method=technique, days_aggregate=rolling_window, m1zero=True, ret_nc_mom=False).to_numpy())
mvsek = np.squeeze(np.array(mvsek))
mvsek = pd.DataFrame(mvsek).T # each column is a path and each row is a moment (mean, variance, skewness, kurtosis)
print(mvsek)
mvsek = mvsek.mean(axis=1) # rowwise means

# modify variance
# rm[1] = rm[1]**2

print(mvsek)

# Calculate cumulants
cumulants = scipy_mvsek_to_cumulants(mean=mvsek[0], variance=mvsek[1], skewness=mvsek[2], excess_kurtosis=mvsek[3])
print(cumulants)

# print("Haozhe MVSK:", scipy_mvsek_to_cumulants(*[-0.01676743,  0.01647651, -0.19187426,  3.36067764-3]))
# print("Haozhe Cumulants", [-0.01401158, 0.01647651, -0.0006112, 0.00093973])

# Expansion methods
x = np.linspace(-1, 1, 1000)
gc = gram_charlier_expansion(x, *cumulants)
ew = edgeworth_expansion(x, *cumulants)
sp = saddlepoint_approximation(x, *cumulants)

# kde of monthly log returns
# log_returns = process_df.values.flatten()
# monthly_log_returns = process_df.rolling(window=rolling_window).sum().dropna().values.flatten()
monthly_log_returns = process_df.resample('M').sum().values.flatten() # why does this work so much better?
kde = gaussian_kde(monthly_log_returns)
kde_values = kde(x)

# Plotting
plt.plot(x, kde_values, 'r--', label='KDE')
plt.plot(x, gc, 'b-', label='Gram-Charlier Expansion')
plt.plot(x, ew, 'g-', label='Edgeworth Expansion')
plt.plot(x, sp, 'y-', label='Saddlepoint Approximation')
plt.title('KDE vs GC, EW, SP of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()