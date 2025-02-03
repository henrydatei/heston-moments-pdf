import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import argparse
import logging
import sqlite3
# from realized_cumulants import r_cumulants

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.SimHestonQE import Heston_QE
from simulation.utils import process_to_log_returns_interday
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_NP_return
from code_from_haozhe.RealizedSkewness_NP_MonthlyOverlap import rCumulants

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(results_dir, 'log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

start_date = '2000-07-01'
end_date = '2100-07-01'
time_points = 15 * 22 * 79 * 12
burnin = 3 * 22 * 79 * 12
T = 15
S0 = 100
paths = 1
seed = 0
np.random.seed(seed)

# Full Simulation
v0_min = 0.01
v0_max = 0.5
v0_step = 0.05
kappa_min = 0.01
kappa_max = 1
kappa_step = 0.1
theta_min = 0.01
theta_max = 1
theta_step = 0.05
sigma_min = 0.01
sigma_max = 1
sigma_step = 0.05
mu_min = -0.1
mu_max = 0.1
mu_step = 0.01
rho_min = -0.9
rho_max = 0.9
rho_step = 0.1

v0s = np.arange(v0_min, v0_max, v0_step)
kappas = np.arange(kappa_min, kappa_max, kappa_step)
thetas = np.arange(theta_min, theta_max, theta_step)
sigmas = np.arange(sigma_min, sigma_max, sigma_step)
mus = np.arange(mu_min, mu_max, mu_step)
rhos = np.arange(rho_min, rho_max, rho_step)

# Short local test
# v0s = [0.21]
# kappas = [0.51]
# thetas = [0.36]
# sigmas = [0.16]
mus = [0.05]
# rhos = [-0.8]

def load_existing_combinations():
    query = """SELECT start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin FROM simulations"""
    
    with sqlite3.connect('simulations.db') as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return set(cursor.fetchall())  # Speichert alle Kombinationen als Tupel in einem Set

def create_simulation_and_save_it(params):
    subjob_id, start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin = params
    worker_id = os.getpid()
    logging.info(f"Worker {worker_id}: Running simulation with v0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, mu={mu}, rho={rho}, seed={seed}")

    subjob_dir = os.path.join(results_dir, f'subjob_{subjob_id}')
    os.makedirs(subjob_dir, exist_ok=True)

    try:
        process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
        # if mu != 0:
        #     process = process - mu
        process_df = process_to_log_returns_interday(process, start_date, end_date)

        max_number_of_same_prices = process_df.iloc[:, 0].value_counts().max()
        
        technique = RM_NP_return
        mvsek = []
        realized_cumulants = []
        for column in process_df.columns:
            mvsek.append(rMoments_mvsek(process_df[column], method=technique, days_aggregate=22, m1zero=False, ret_nc_mom=True).to_numpy())
            realized_cumulants.append(rCumulants(process_df[column], method=technique, months_overlap=6).to_numpy())
            # values = process_df[column].to_numpy(dtype=np.float64)
            # timestamps = process_df.index.to_numpy(dtype=np.int64) // 10 ** 9
            # cumulants = r_cumulants(values, timestamps, months_overlap=6)
            # realized_cumulants.append(cumulants)
            # logging.info(f"Worker {worker_id}: Results from Rust Code: {cumulants}")

        mvsek = pd.DataFrame(np.squeeze(np.array(mvsek))).T.mean(axis=1)
        realized_cumulants = pd.DataFrame(np.squeeze(np.array(realized_cumulants))).T.mean(axis=1)

        result = (
            mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths,
            start_date, end_date, 'day', max_number_of_same_prices,
            realized_cumulants[0], realized_cumulants[1], realized_cumulants[2], realized_cumulants[3],
            mvsek[0], mvsek[1], mvsek[2], mvsek[3], mvsek[4], mvsek[5], seed
        )

        filename = os.path.join(subjob_dir, f'simulation_{worker_id}.csv')
        df = pd.DataFrame([result])
        df.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))

        logging.info(f"Worker {worker_id}: Simulation saved in {filename}")
        
        logging.info(f"Worker {worker_id} result: {result}")

    except Exception as e:
        logging.error(f"Worker {worker_id}: Error {e}")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Run a subset of simulations.')
    parser.add_argument('--i', type=int, required=True, help='Index of the job')
    parser.add_argument('--chunks', type=int, default=1, help='Number of chunks')
    args = parser.parse_args()

    i = args.i
    num_chunks = args.chunks
    
    existing_combinations = load_existing_combinations()

    parameter_list = [
        (i, start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin)
        for v0 in v0s for kappa in kappas for theta in thetas
        for sigma in sigmas for mu in mus for rho in rhos
    ]

    chunk_size = len(parameter_list) // num_chunks
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size if i < num_chunks - 1 else len(parameter_list)

    sub_parameter_list = parameter_list[start_index:end_index]

    print(f"Processing chunk {i}: {len(sub_parameter_list)} simulations.")
    
    # Entferne bereits berechnete Parameterkombinationen durch schnellen Lookup im Set
    sub_parameter_list = [params for params in sub_parameter_list if params[1:] not in existing_combinations]
    
    print(f"Processing chunk {i}: {len(sub_parameter_list)} simulations left after removing already computed results.")

    # Nutzung aller verfügbaren CPU-Kerne
    with Pool(processes=cpu_count()) as pool:
        pool.map(create_simulation_and_save_it, sub_parameter_list)

    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
