import numpy as np
import sys
import os
import sqlite3
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expansion_methods.all_methods import normal_expansion
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate
from compare_distributions.distances import pdf_to_cdf, KS_test_sample_from_cdf, Cramer_von_Mises_test, Andersen_Darling_test
from simulation_database.database_utils import update_multiple_values, add_column

def do_tests(expansion_method, argument, x, x_theory, density):
    expansion = expansion_method(x, *argument)

    # CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
    theory_cdf = pdf_to_cdf(x_theory, density)
    empirical_cdf = pdf_to_cdf(x, expansion)

    ks_statistic, ks_p_value = KS_test_sample_from_cdf(x_theory, theory_cdf, x, empirical_cdf)

    # cv_statistic, cv_p_value = Cramer_von_Mises_test(x_theory, theory_cdf, x, empirical_cdf)
    
    # ad_statistic, ad_p_value = Andersen_Darling_test(x_theory, theory_cdf, x, empirical_cdf)
    
    cv_statistic, cv_p_value, ad_statistic, ad_p_value = None, None, None, None
    return ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value

def process_simulation(simulation):
    cumulants = simulation[16:18]
    mu = simulation[1]
    kappa = simulation[2]
    theta = simulation[3]
    sigma = simulation[4]
    rho = simulation[5]
    v0 = simulation[6]
    
    results = {}
    
    # Theoretical density
    try:
        x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)
    except Exception as e:
        logging.error(f'Error with theoretical density for simulation {simulation[0]}: {e}')
        return
    x = np.linspace(-2, 2, 1000)

    # GC with cumulants
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(normal_expansion, cumulants, x, x_theory, density)
        results['NO_cum_KS_stat'] = ks_statistic
        results['NO_cum_KS_p'] = ks_p_value
        # results['NO_cum_CV_stat'] = cv_statistic
        # results['NO_cum_CV_p'] = cv_p_value
        # results['NO_cum_AD_stat'] = ad_statistic
        # results['NO_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with Normal with cumulants for simulation {simulation[0]}: {e}')
        
    # write results to database
    update_multiple_values('simulations', results, mu, kappa, theta, sigma, rho, v0)
    
def main():
    add_column('simulations', 'NO_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'NO_cum_KS_p', 'FLOAT')
    add_column('simulations', 'NO_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'NO_cum_CV_p', 'FLOAT')
    add_column('simulations', 'NO_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'NO_cum_AD_p', 'FLOAT')
    
    conn = sqlite3.connect('simulations.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM simulations WHERE NO_cum_KS_stat IS NULL AND feller_condition = 1')
    simulations = cursor.fetchall()

    for simulation in tqdm(simulations):
        process_simulation(simulation)
    
    conn.close()
    
if __name__ == '__main__':
    main()