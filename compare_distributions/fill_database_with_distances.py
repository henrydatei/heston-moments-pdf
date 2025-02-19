import numpy as np
import sys
import os
import sqlite3
import logging
import argparse
from multiprocessing import Pool
import pandas as pd
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expansion_methods.all_methods import moments_to_cumulants, moments_to_mvsek, cumulants_to_mvsek, gram_charlier_expansion, gram_charlier_expansion_positivity_constraint, edgeworth_expansion, edgeworth_expansion_positivity_constraint, cornish_fisher_expansion, saddlepoint_approximation
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate
from compare_distributions.distances import pdf_to_cdf, KS_test_sample_from_cdf, Cramer_von_Mises_test, Andersen_Darling_test

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(results_dir, 'log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def do_tests(expansion_method, argument, fakasawa, x, x_theory, density):
    if fakasawa:
        expansion = expansion_method(x, *argument, fakasawa=True)
    else:
        expansion = expansion_method(x, *argument)

    # CDFs + Normalisation (since the sum of the PDF over all space should equal 1)
    theory_cdf = pdf_to_cdf(x_theory, density)
    empirical_cdf = pdf_to_cdf(x, expansion)

    ks_statistic, ks_p_value = KS_test_sample_from_cdf(x_theory, theory_cdf, x, empirical_cdf)

    cv_statistic, cv_p_value = Cramer_von_Mises_test(x_theory, theory_cdf, x, empirical_cdf)
    
    ad_statistic, ad_p_value = Andersen_Darling_test(x_theory, theory_cdf, x, empirical_cdf)
    
    return ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value
    
def process_simulation(simulation):
    cumulants = simulation[16:20]
    moments = simulation[20:24]
    mu = simulation[1]
    kappa = simulation[2]
    theta = simulation[3]
    sigma = simulation[4]
    rho = simulation[5]
    v0 = simulation[6]
    
    results = {
        'mu': mu,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'rho': rho,
        'v0': v0
    }
    
    # Theoretical density
    try:
        x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)
    except Exception as e:
        logging.error(f'Error with theoretical density for simulation {simulation[0]}: {e}')
        return
    x = np.linspace(-2, 2, 1000)

    # GC with cumulants
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(gram_charlier_expansion, cumulants, True, x, x_theory, density)
        results['GC_cum_KS_stat'] = ks_statistic
        results['GC_cum_KS_p'] = ks_p_value
        results['GC_cum_CV_stat'] = cv_statistic
        results['GC_cum_CV_p'] = cv_p_value
        results['GC_cum_AD_stat'] = ad_statistic
        results['GC_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with GC with cumulants for simulation {simulation[0]}: {e}')
    
    # GC with moments
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(gram_charlier_expansion, moments_to_cumulants(*moments), False, x, x_theory, density)
        results['GC_mom_KS_stat'] = ks_statistic
        results['GC_mom_KS_p'] = ks_p_value
        results['GC_mom_CV_stat'] = cv_statistic
        results['GC_mom_CV_p'] = cv_p_value
        results['GC_mom_AD_stat'] = ad_statistic
        results['GC_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with GC with moments for simulation {simulation[0]}: {e}')
    
    # GC with cumulants, positivity
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(gram_charlier_expansion_positivity_constraint, cumulants_to_mvsek(*cumulants), True, x, x_theory, density)
        results['GC_pos_cum_KS_stat'] = ks_statistic
        results['GC_pos_cum_KS_p'] = ks_p_value
        results['GC_pos_cum_CV_stat'] = cv_statistic
        results['GC_pos_cum_CV_p'] = cv_p_value
        results['GC_pos_cum_AD_stat'] = ad_statistic
        results['GC_pos_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with GC with cumulants, positivity for simulation {simulation[0]}: {e}')
    
    # GC with moments, positivity
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(gram_charlier_expansion_positivity_constraint, moments_to_mvsek(*moments), False, x, x_theory, density)
        results['GC_pos_mom_KS_stat'] = ks_statistic
        results['GC_pos_mom_KS_p'] = ks_p_value
        results['GC_pos_mom_CV_stat'] = cv_statistic
        results['GC_pos_mom_CV_p'] = cv_p_value
        results['GC_pos_mom_AD_stat'] = ad_statistic
        results['GC_pos_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with GC with moments, positivity for simulation {simulation[0]}: {e}')
    
    # EW with cumulants
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(edgeworth_expansion, cumulants, True, x, x_theory, density)
        results['EW_cum_KS_stat'] = ks_statistic
        results['EW_cum_KS_p'] = ks_p_value
        results['EW_cum_CV_stat'] = cv_statistic
        results['EW_cum_CV_p'] = cv_p_value
        results['EW_cum_AD_stat'] = ad_statistic
        results['EW_cum_AD_p'] = ad_p_value
    except Exception as e:
        print(f'Error with EW with cumulants for simulation {simulation[0]}: {e}')
    
    # EW with moments
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(edgeworth_expansion, moments_to_cumulants(*moments), False, x, x_theory, density)
        results['EW_mom_KS_stat'] = ks_statistic
        results['EW_mom_KS_p'] = ks_p_value
        results['EW_mom_CV_stat'] = cv_statistic
        results['EW_mom_CV_p'] = cv_p_value
        results['EW_mom_AD_stat'] = ad_statistic
        results['EW_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with EW with moments for simulation {simulation[0]}: {e}')
    
    # EW with cumulants, positivity
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(edgeworth_expansion_positivity_constraint, cumulants_to_mvsek(*cumulants), True, x, x_theory, density)
        results['EW_pos_cum_KS_stat'] = ks_statistic
        results['EW_pos_cum_KS_p'] = ks_p_value
        results['EW_pos_cum_CV_stat'] = cv_statistic
        results['EW_pos_cum_CV_p'] = cv_p_value
        results['EW_pos_cum_AD_stat'] = ad_statistic
        results['EW_pos_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with EW with cumulants, positivity for simulation {simulation[0]}: {e}')
    
    # EW with moments, positivity
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(edgeworth_expansion_positivity_constraint, moments_to_mvsek(*moments), False, x, x_theory, density)
        results['EW_pos_mom_KS_stat'] = ks_statistic
        results['EW_pos_mom_KS_p'] = ks_p_value
        results['EW_pos_mom_CV_stat'] = cv_statistic
        results['EW_pos_mom_CV_p'] = cv_p_value
        results['EW_pos_mom_AD_stat'] = ad_statistic
        results['EW_pos_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with EW with moments, positivity for simulation {simulation[0]}: {e}')
    
    # CF with cumulants
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(cornish_fisher_expansion, cumulants_to_mvsek(*cumulants), False, x, x_theory, density)
        results['CF_cum_KS_stat'] = ks_statistic
        results['CF_cum_KS_p'] = ks_p_value
        results['CF_cum_CV_stat'] = cv_statistic
        results['CF_cum_CV_p'] = cv_p_value
        results['CF_cum_AD_stat'] = ad_statistic
        results['CF_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with CF with cumulants for simulation {simulation[0]}: {e}')
    
    # CF with moments
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(cornish_fisher_expansion, moments_to_mvsek(*moments), False, x, x_theory, density)
        results['CF_mom_KS_stat'] = ks_statistic
        results['CF_mom_KS_p'] = ks_p_value
        results['CF_mom_CV_stat'] = cv_statistic
        results['CF_mom_CV_p'] = cv_p_value
        results['CF_mom_AD_stat'] = ad_statistic
        results['CF_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with CF with moments for simulation {simulation[0]}: {e}')
    
    # SP with cumulants
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(saddlepoint_approximation, cumulants, False, x, x_theory, density)
        results['SP_cum_KS_stat'] = ks_statistic
        results['SP_cum_KS_p'] = ks_p_value
        results['SP_cum_CV_stat'] = cv_statistic
        results['SP_cum_CV_p'] = cv_p_value
        results['SP_cum_AD_stat'] = ad_statistic
        results['SP_cum_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with SP with cumulants for simulation {simulation[0]}: {e}')
    
    # SP with moments
    try:
        ks_statistic, ks_p_value, cv_statistic, cv_p_value, ad_statistic, ad_p_value = do_tests(saddlepoint_approximation, moments_to_cumulants(*moments), False, x, x_theory, density)
        results['SP_mom_KS_stat'] = ks_statistic
        results['SP_mom_KS_p'] = ks_p_value
        results['SP_mom_CV_stat'] = cv_statistic
        results['SP_mom_CV_p'] = cv_p_value
        results['SP_mom_AD_stat'] = ad_statistic
        results['SP_mom_AD_p'] = ad_p_value
    except Exception as e:
        logging.error(f'Error with SP with moments for simulation {simulation[0]}: {e}')
        
    # write results to csv
    df = pd.DataFrame(results, index=[0])
    filename = os.path.join(results_dir, f"results_{os.getpid()}.csv")
    df.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, help="Aktueller Chunk-Index", default=0)
    parser.add_argument("--chunks", type=int, help="Gesamtanzahl der Chunks", default=1000000)
    args = parser.parse_args()
    
    c = sqlite3.connect(f'simulations_{args.i}.db')
    cursor = c.cursor()

    total_rows = c.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
    rows_per_chunk = math.ceil(total_rows / args.chunks)

    start_index = args.i * rows_per_chunk
    end_index = min(start_index + rows_per_chunk, total_rows)

    query = f"SELECT * FROM simulations LIMIT {rows_per_chunk} OFFSET {start_index}"
    simulations = cursor.execute(query).fetchall()
    c.close()
    
    logging.info(f"Processing chunk {args.i}: {len(simulations)} simulations.")
    logging.info(f"Total rows: {total_rows}, rows per chunk: {rows_per_chunk}, Query: {query}")
    
    results_dir = os.path.join(results_dir, f"task_{args.i}")
    os.makedirs(results_dir, exist_ok=True)
    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(process_simulation, simulations)