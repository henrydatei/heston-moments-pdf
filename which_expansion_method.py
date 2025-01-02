import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.utils import process_to_log_returns_interday, process_to_log_returns
from simulation.SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_CL, RM_NP, RM_ORS, RM_ACJV, RM_NP_return
from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion, edgeworth_expansion, saddlepoint_approximation
from code_from_haozhe.GramCharlier_expansion import Expansion_GramCharlier as gram_charlier_expansion_haozhe
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate

np.random.seed(0)

# Define parameters for the simulation
# time_points = 3 * 12 * 22 * 79  # 3 years with 12 months with 22 trading days with 79 prices (5 minute intervals from 9:30 am to 4pm and inital price
time_points = 60 * 22 * 12
start_date = '2014-07-01'
end_date = '2076-07-01'
T = 60
S0 = 100
paths = 1
v0 = 0.19
kappa = 3
theta = 0.19
sigma = 0.4
mu = 0 # martingale
rho = -0.7
rolling_window = 22

# Eraker 2004 (for time unit = 1 day)
# mu = 0.026 # is not 0, so we need to de-mean the data
# theta = 1.933
# kappa = 0.019
# rho = -0.569
# sigma = 0.220

# Eraker 2004 (for time unit = 5 min)
# kappa = 0.019 * 1/78
# sigma = 0.220 * np.sqrt(1/78)

# Check if Feller condition is satisfied
print(f'Feller condition: {2 * kappa * theta > sigma**2}')

# Simulation
process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
if mu != 0:
    # de-mean the data
    process = process - mu
# process_df = process_to_log_returns_interday(process, start_date, end_date)
process_df = process_to_log_returns(process, start_date, end_date, time_points)

# Estimate moments
technique = RM_NP_return
mvsek = []
for column in process_df.columns:
    mvsek.append(rMoments_mvsek(process_df[column], method=technique, days_aggregate=rolling_window, m1zero=True, ret_nc_mom=True).to_numpy())
mvsek = np.squeeze(np.array(mvsek))
mvsek = pd.DataFrame(mvsek).T # each column is a path and each row is a moment (mean, variance, 3rd moment, 4th moment, skewness, excess kurtosis)
print(mvsek)
mvsek = mvsek.mean(axis=1) # rowwise means

# modify variance
# rm[1] = rm[1]**2

print(mvsek)

# Calculate cumulants
cumulants = scipy_mvsek_to_cumulants(mean=mvsek[0], variance=mvsek[1], skewness=mvsek[4], excess_kurtosis=mvsek[5])
print(cumulants)

# print("Haozhe MVSK:", scipy_mvsek_to_cumulants(*[-0.01676743,  0.01647651, -0.19187426,  3.36067764-3]))
# print("Haozhe Cumulants", [-0.01401158, 0.01647651, -0.0006112, 0.00093973])

# Cumulants = Central moments by Fukasawa (2021), since drift = 0 (martingale): Central moments = moments
cumulants2 = (mvsek[0], mvsek[1], mvsek[2], mvsek[3])
print(cumulants2)

cumulants_true = (-0.00791667, 0.01601168, -0.00056375, 0.00088632)
cumulants_haozhe = (-0.01401158, 0.01647651, -0.0006112, 0.00093973)

# Expansion methods
x = np.linspace(-2, 2, 1000)
gc = gram_charlier_expansion(x, *cumulants)
gc2 = gram_charlier_expansion(x, *cumulants2)
gc_true = gram_charlier_expansion(x, *cumulants_true)
gc_haozhe = gram_charlier_expansion(x, *cumulants_haozhe)

haozhe_gc = gram_charlier_expansion_haozhe(cumulants)
haozhe_gc2 = gram_charlier_expansion_haozhe(cumulants2)
haozhe_gc_true = gram_charlier_expansion_haozhe(cumulants_true)
haozhe_gc_haozhe = gram_charlier_expansion_haozhe(cumulants_haozhe)

ew = edgeworth_expansion(x, *cumulants)
sp = saddlepoint_approximation(x, *cumulants)

# Density
x_density, density = compute_density_via_ifft_accurate(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, tau=1/12)

# kde of monthly log returns
# log_returns = process_df.values.flatten()
# monthly_log_returns = process_df.rolling(window=rolling_window).sum().dropna().values.flatten()
monthly_log_returns = process_df.resample('M').sum().values.flatten() # why does this work so much better?
kde = gaussian_kde(monthly_log_returns)
kde_values = kde(x)

# Plotting
plt.plot(x_density, density, 'y-', label='Theoretical Density')
# plt.plot(x, kde_values, 'y--', label='KDE')
# plt.plot(x, gc, 'b-', label='GC Cumulants calculated from NP Moments')
# plt.plot(x, gc2, 'r-', label='GC Cumulants = Central Moments')
# plt.plot(x, gc_true, 'm-', label='GC True Cumulants')
# plt.plot(x, gc_haozhe, 'g-', label='GC Haozhe Cumulants (realized cumulant method)')

plt.plot(x, haozhe_gc, 'b--', label='HaozheGC Cumulants calculated from NP Moments')
plt.plot(x, haozhe_gc2, 'r--', label='HaozheGC Cumulants = Central Moments')
plt.plot(x, haozhe_gc_true, 'm--', label='HaozheGC True Cumulants')
plt.plot(x, haozhe_gc_haozhe, 'g--', label='HaozheGC Haozhe Cumulants (realized cumulant method)')

# plt.plot(x, ew, 'g-', label='Edgeworth Expansion')
# plt.plot(x, sp, 'y-', label='Saddlepoint Approximation')
# plt.title('KDE vs GC, EW, SP of Log-Returns')
plt.legend()

# plt.tight_layout()
plt.ylim(0, 3.5)
plt.xlim(-2, 2)
plt.show()