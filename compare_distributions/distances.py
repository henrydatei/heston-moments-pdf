from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.utils import process_to_log_returns
from simulation.SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_CL, RM_NP, RM_ORS, RM_ACJV, RM_NP_return
from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate

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

# # Eraker 2004 (for time unit = 1 day)
# mu = 0.026 # is not 0, so we need to de-mean the data
# theta = 1.933
# kappa = 0.019
# rho = -0.569
# sigma = 0.220

# # Eraker 2004 (for time unit = 5 min)
# kappa = 0.019 * 1/78
# sigma = 0.220 * np.sqrt(1/78)

# Check if Feller condition is satisfied
print(f'Feller condition: {2 * kappa * theta > sigma**2}')

# Simulation
process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
process_df = process_to_log_returns(process, start_date, end_date)

# Estimate moments
technique = RM_NP_return
mvsek = []
for column in process_df.columns:
    mvsek.append(rMoments_mvsek(process_df[column], method=technique, days_aggregate=rolling_window, m1zero=True, ret_nc_mom=False).to_numpy())
mvsek = np.squeeze(np.array(mvsek))
mvsek = pd.DataFrame(mvsek).T # each column is a path and each row is a moment (mean, variance, skewness, kurtosis)
print(mvsek)
mvsek = mvsek.mean(axis=1) # rowwise means

print(mvsek)

# Calculate cumulants
cumulants = scipy_mvsek_to_cumulants(mean=mvsek[0], variance=mvsek[1], skewness=mvsek[2], excess_kurtosis=mvsek[3])
print(cumulants)

# Expansion methods
x = np.linspace(-2, 2, 1000)
gc = gram_charlier_expansion(x, *cumulants)

# Theoretical density
x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)

# Plotting
plt.plot(x_theory, density, 'r--', label='Theoretical Density')
plt.plot(x, gc, 'b-', label='Gram-Charlier Expansion')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

# CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
dx_empirical = x[1] - x[0]
dx_theory = x_theory[1] - x_theory[0]
empirical_cdf = np.cumsum(gc) * dx_empirical
theory_cdf = np.cumsum(density) * dx_theory
empirical_cdf = empirical_cdf / empirical_cdf[-1]
theory_cdf = theory_cdf / theory_cdf[-1]

# Plotting
plt.plot(x_theory, theory_cdf, 'r--', label='Theoretical CDF')
plt.plot(x, empirical_cdf, 'b-', label='Gram-Charlier Expansion CDF')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

ks_statistic, p_value = stats.ks_2samp(empirical_cdf, theory_cdf)
print(f'KS Statistic: {ks_statistic}, P-Value: {p_value}')