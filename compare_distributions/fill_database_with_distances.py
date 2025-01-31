import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sqlite3
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate
from simulation_database.database_utils import add_column
from compare_distributions.distances import pdf_to_cdf, KS_test_sample_from_pdf, KS_test_sample_from_cdf, Cramer_von_Mises_test

from code_from_haozhe.GramCharlier_expansion import Expansion_GramCharlier

c = sqlite3.connect('simulations.db')
cursor = c.cursor()

# get all simulations and store into a list
simulations = cursor.execute('SELECT * FROM simulations LIMIT 2').fetchall()
c.close()

for simulation in tqdm(simulations):
    print(simulation)

    cumulants = simulation[16:20]
    mu = simulation[1]
    kappa = simulation[2]
    theta = simulation[3]
    sigma = simulation[4]
    rho = simulation[5]

    print(f'Random Simulation {simulation[0]} Cumulants: {cumulants}')
    print(f'Random Simulation Parameters: {mu, kappa, theta, sigma, rho}')

    # Expansion methods
    x = np.linspace(-2, 2, 1000)
    gc = gram_charlier_expansion(x, *cumulants, fakasawa=True)
    gc_haozhe = Expansion_GramCharlier(cumulants)

    # true_cumulant = np.array([-0.00791667, 0.01601168, -0.00056375, 0.00088632])
    # gc_true = gram_charlier_expansion(x, *true_cumulant)
    # gc_true_haozhe = Expansion_GramCharlier(true_cumulant)

    # Theoretical density
    x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)

    # CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
    theory_cdf = pdf_to_cdf(x_theory, density)
    empirical_cdf = pdf_to_cdf(x, gc)
    haozhe_cdf = pdf_to_cdf(x, gc_haozhe)

    # ks_statistic, p_value = KS_test_sample_from_pdf(x_theory, density, x, gc)
    # print(f'Henry PDF KS Statistic: {ks_statistic}, P-Value: {p_value}')

    ks_statistic, p_value = KS_test_sample_from_cdf(x_theory, theory_cdf, x, empirical_cdf)
    print(f'KS Statistic: {ks_statistic}, P-Value: {p_value}')

    cv_statistic, p_value = Cramer_von_Mises_test(x_theory, theory_cdf, x, empirical_cdf)
    print(f' Cramer von Mises Statistic: {cv_statistic}, P-Value: {p_value}')