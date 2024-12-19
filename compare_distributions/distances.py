from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.utils import process_to_log_returns
from simulation.SimHestonQE import Heston_QE
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_CL, RM_NP, RM_ORS, RM_ACJV, RM_NP_return
from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate

from code_from_haozhe.GramCharlier_expansion import Expansion_GramCharlier

def pdf_to_cdf(x, pdf, normalize=True):
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf) * dx
    if normalize:
        cdf = cdf / cdf[-1]
    return cdf

def sample_from_pdf(x, pdf, n_samples):
    np.random.seed(0)
    return np.random.choice(x, size=n_samples, p=pdf/np.sum(pdf))

def sample_from_cdf(x, cdf, n_samples):
    np.random.seed(0)
    inv_cdf = interp1d(cdf, x, kind='linear', fill_value='extrapolate')
    uniform_samples = np.random.uniform(0, 1, n_samples)
    return inv_cdf(uniform_samples)

def KS_test(x_1, pdf_1, x_2, pdf_2):
    samples_1 = sample_from_pdf(x_1, pdf_1, 1000)
    samples_2 = sample_from_pdf(x_2, pdf_2, 1000)

    ks_statistic, p_value = stats.ks_2samp(samples_1, samples_2)
    return ks_statistic, p_value

def KS_test_2(x_1, cdf_1, x_2, cdf_2):
    samples_1 = sample_from_cdf(x_1, cdf_1, 1000)
    samples_2 = sample_from_cdf(x_2, cdf_2, 1000)

    ks_statistic, p_value = stats.ks_2samp(samples_1, samples_2)
    return ks_statistic, p_value

def Cramer_von_Mises_test(x_1, pdf_1, x_2, pdf_2):
    samples_1 = sample_from_pdf(x_1, pdf_1, 1000)
    samples_2 = sample_from_pdf(x_2, pdf_2, 1000)

    res = stats.cramervonmises_2samp(samples_1, samples_2)
    return res.statistic, res.pvalue

np.random.seed(0)

# Define parameters for the simulation
time_points = 3 * 12 * 22 * 79  # 3 years with 12 months with 22 trading days with 79 prices (5 minute intervals from 9:30 am to 4pm and inital price
time_points = 60 * 22 * 12
start_date = '2014-07-01'
end_date = '2076-07-01'
T = 3
T = 60
S0 = 100
paths = 10
paths = 1
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
process_df = process_to_log_returns(process, start_date, end_date, time_points, burnin_timesteps=10*22*12)

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
gc_haozhe = Expansion_GramCharlier(cumulants)

true_cumulant = np.array([-0.00791667, 0.01601168, -0.00056375, 0.00088632])
gc_true = gram_charlier_expansion(x, *true_cumulant)
gc_true_haozhe = Expansion_GramCharlier(true_cumulant)

# Theoretical density
x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)

# Plotting
plt.plot(x_theory, density, 'r--', label='Theoretical Density')
plt.plot(x, gc, 'b-', label='Henry Gram-Charlier Expansion')
plt.plot(x, gc_haozhe, 'm-', label='Haozhe Gram-Charlier Expansion')
plt.plot(x, gc_true, 'g-', label='Henry True Gram-Charlier Expansion')
plt.plot(x, gc_true_haozhe, 'y-', label='Haozhe True Gram-Charlier Expansion')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

# CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
theory_cdf = pdf_to_cdf(x_theory, density)
empirical_cdf = pdf_to_cdf(x, gc)

# Plotting
plt.plot(x_theory, theory_cdf, 'r--', label='Theoretical CDF')
plt.plot(x, empirical_cdf, 'b-', label='Gram-Charlier Expansion CDF')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

ks_statistic, p_value = KS_test(x_theory, density, x, gc)
print(f'KS Statistic: {ks_statistic}, P-Value: {p_value}')

ks_statistic, p_value = KS_test_2(x_theory, theory_cdf, x, empirical_cdf)
print(f'KS Statistic: {ks_statistic}, P-Value: {p_value}')

cv_statistic, p_value = Cramer_von_Mises_test(x_theory, density, x, gc)
print(f'Cramer von Mises Statistic: {cv_statistic}, P-Value: {p_value}')