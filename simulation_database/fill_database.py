import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.SimHestonQE import Heston_QE
from simulation.utils import process_to_log_returns_interday, process_to_log_returns

np.random.seed(33)
simulations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations')
conn = sqlite3.connect('simulation_database.db')
c = conn.cursor()

start_date = '2000-07-01'
end_date = '2100-07-01'
time_points = 60 * 22 * 12
burnin = 10 * 22 * 12
T = 60
S0 = 100
paths = 1
# v0 = 0.19
# kappa = 3
# theta = 0.19
# sigma = 0.4
# mu = 0 # martingale
# rho = -0.7

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

# Test Simulation
v0_min = 0.01
v0_max = 0.5
v0_step = 0.2
kappa_min = 0.01
kappa_max = 1
kappa_step = 0.5
theta_min = 0.01
theta_max = 1
theta_step = 0.3
sigma_min = 0.01
sigma_max = 1
sigma_step = 0.3
mu_min = -0.1
mu_max = 0.1
mu_step = 0.01
rho_min = -0.9
rho_max = 0.9
rho_step = 0.3

v0s = np.arange(v0_min, v0_max, v0_step)
kappas = np.arange(kappa_min, kappa_max, kappa_step)
thetas = np.arange(theta_min, theta_max, theta_step)
sigmas = np.arange(sigma_min, sigma_max, sigma_step)
mus = np.arange(mu_min, mu_max, mu_step)
rhos = np.arange(rho_min, rho_max, rho_step)

# v0s = [0.1, 0.5]
# kappas = [0.5, 3]
# thetas = [0.01, 0.5]
# sigmas = [0.1, 0.5]
# mus = [0]
# rhos = [0.3, -0.7]

print(len(v0s), len(kappas), len(thetas), len(sigmas), len(mus), len(rhos))
print(len(v0s) * len(kappas) * len(thetas) * len(sigmas) * len(mus) * len(rhos))
print(v0s)
print(kappas)
print(thetas)
print(sigmas)
print(mus)
print(rhos)

def create_simulation_and_save_it(start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin):
    process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
    if mu != 0:
        # de-mean the data
        process = process - mu
    # process_df = process_to_log_returns_interday(process, start_date, end_date)
    process_df = process_to_log_returns(process, start_date, end_date, time_points, burnin)
    
    # number of equal values in the first column -> indicator for maleformed parameters when feller condition is far away from saisfied
    max_number_of_same_prices = process_df.iloc[:, 0].value_counts().max()

    # get id of last simulation
    c.execute('SELECT id FROM simulations ORDER BY id DESC LIMIT 1')
    last_id = c.fetchone()
    if last_id is None:
        last_id = 0
    else:
        last_id = last_id[0]

    # insert parameters into database and return the id
    filename_csv = f'simulation_{last_id + 1}.csv'
    filename_gz = f'simulation_{last_id + 1}.csv.gz'
    c.execute('INSERT INTO simulations (id, mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, start_date, end_date, interval, filename, max_number_of_same_prices) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (last_id + 1, mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, start_date, end_date, 'day', filename_gz, max_number_of_same_prices))
    conn.commit()

    # save process_df to file and compress it
    process_df.to_csv(os.path.join(simulations_dir, filename_csv), index=False)

    # compress the file
    os.system(f'gzip {os.path.join(simulations_dir, filename_csv)}')
    
for v0 in tqdm(v0s):
    for kappa in tqdm(kappas, leave=False):
        for theta in tqdm(thetas, leave=False):
            for sigma in tqdm(sigmas, leave=False):
                for mu in tqdm(mus, leave=False):
                    for rho in tqdm(rhos, leave=False):
                        create_simulation_and_save_it(start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin)
    