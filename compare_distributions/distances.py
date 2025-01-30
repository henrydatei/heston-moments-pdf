from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import sqlite3

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

c = sqlite3.connect('simulations.db')
cursor = c.cursor()
random_simulation = cursor.execute('SELECT * FROM simulations WHERE max_number_of_same_prices < 10 ORDER BY RANDOM() LIMIT 1').fetchone()
c.close()

cumulants = random_simulation[16:20]
mu = random_simulation[1]
kappa = random_simulation[2]
theta = random_simulation[3]
sigma = random_simulation[4]
rho = random_simulation[5]

print(f'Random Simulation {random_simulation[0]} Cumulants: {cumulants}')
print(f'Random Simulation Parameters: {mu, kappa, theta, sigma, rho}')

# Expansion methods
x = np.linspace(-2, 2, 1000)
gc = gram_charlier_expansion(x, *cumulants, fakasawa=True)
gc_haozhe = Expansion_GramCharlier(cumulants)

true_cumulant = np.array([-0.00791667, 0.01601168, -0.00056375, 0.00088632])
gc_true = gram_charlier_expansion(x, *true_cumulant)
gc_true_haozhe = Expansion_GramCharlier(true_cumulant)

# Theoretical density
x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)

# Plotting
plt.plot(x_theory, density, 'r--', label='Theoretical Density')
plt.plot(x, gc, 'b-', label='Gram-Charlier Expansion')
# plt.plot(x, gc_haozhe, 'm-', label='Haozhe Gram-Charlier Expansion')
plt.plot(x, gc_true, 'g-', label='True Gram-Charlier Expansion')
# plt.plot(x, gc_true_haozhe, 'y-', label='Haozhe True Gram-Charlier Expansion')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

# CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
theory_cdf = pdf_to_cdf(x_theory, density)
empirical_cdf = pdf_to_cdf(x, gc)
haozhe_cdf = pdf_to_cdf(x, gc_haozhe)

# Plotting
plt.plot(x_theory, theory_cdf, 'r--', label='Theoretical CDF')
plt.plot(x, empirical_cdf, 'b-', label='Gram-Charlier Expansion CDF')
# plt.plot(x, haozhe_cdf, 'm-', label='Haozhe Gram-Charlier Expansion CDF')
plt.title('Theoretical vs GC of Log-Returns')
plt.legend()

plt.tight_layout()
plt.show()

ks_statistic, p_value = KS_test(x_theory, density, x, gc)
print(f'Henry PDF KS Statistic: {ks_statistic}, P-Value: {p_value}')

ks_statistic, p_value = KS_test_2(x_theory, theory_cdf, x, empirical_cdf)
print(f'Henry CDF KS Statistic: {ks_statistic}, P-Value: {p_value}')

cv_statistic, p_value = Cramer_von_Mises_test(x_theory, density, x, gc)
print(f'Henry PDF Cramer von Mises Statistic: {cv_statistic}, P-Value: {p_value}')

# ks_statistic, p_value = KS_test(x_theory, density, x, gc_haozhe)
# print(f'Haozhe PDF KS Statistic: {ks_statistic}, P-Value: {p_value}')

# cv_statistic, p_value = Cramer_von_Mises_test(x_theory, density, x, gc_haozhe)
# print(f'Haozhe PDF Cramer von Mises Statistic: {cv_statistic}, P-Value: {p_value}')

# print(f"Henry JS Divergence: {jensenshannon(density, gc)}")
# print(f"Haozhe JS Divergence: {jensenshannon(density, gc_haozhe)}")