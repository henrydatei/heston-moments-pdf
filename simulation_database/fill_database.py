import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.SimHestonQE import Heston_QE
from simulation.utils import process_to_log_returns_interday, process_to_log_returns
from code_from_haozhe.RealizedMomentsEstimator_Aggregate_update import rMoments_mvsek, RM_NP_return
from code_from_haozhe.RealizedSkewness_NP_MonthlyOverlap import rCumulants

np.random.seed(33)
simulations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations')
# conn = sqlite3.connect('simulation_database.db')
# c = conn.cursor()

start_date = '2000-07-01'
end_date = '2100-07-01'
# time_points = 60 * 22 * 12
time_points = 15 * 22 * 79 * 12
# burnin = 10 * 22 * 12
burnin = 3 * 22 * 79 * 12
# T = 60
T = 15
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

v0s = [0.1]
kappas = [3]
thetas = [0.01, 0.5]
sigmas = [0.1, 0.5]
mus = [0]
rhos = [-0.7]

print(len(v0s), len(kappas), len(thetas), len(sigmas), len(mus), len(rhos))
print(v0s)
print(kappas)
print(thetas)
print(sigmas)
print(mus)
print(rhos)

def create_simulation_and_save_it(conn, start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin):
    
    c = conn.cursor()
    
    # check if this parameter combination is already in the database
    c.execute('SELECT * FROM simulations WHERE v0 = ? AND kappa = ? AND theta = ? AND sigma = ? AND mu = ? AND rho = ? AND time_points = ? AND burnin = ? AND T = ? AND S0 = ? AND paths = ? AND start_date = ? AND end_date = ?', (v0, kappa, theta, sigma, mu, rho, time_points, burnin, T, S0, paths, start_date, end_date))
    if c.fetchone() is not None:
        print(f'Simulation with parameters v0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, mu={mu}, rho={rho}, time_points={time_points}, burnin={burnin}, T={T}, S0={S0}, paths={paths}, start_date={start_date}, end_date={end_date} already exists.')
        return
    
    process = Heston_QE(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, mu=mu, rho=rho, T=T, N=time_points, n_paths=paths)
    if mu != 0:
        # de-mean the data
        process = process - mu
    process_df = process_to_log_returns_interday(process, start_date, end_date)
    # process_df = process_to_log_returns(process, start_date, end_date, time_points, burnin)
    
    # number of equal values in the first column -> indicator for maleformed parameters when feller condition is far away from saisfied
    max_number_of_same_prices = process_df.iloc[:, 0].value_counts().max()
    
    # estimate moments and cumulants
    technique = RM_NP_return
    mvsek = []
    realized_cumulants = []
    for column in process_df.columns:
        mvsek.append(rMoments_mvsek(process_df[column], method=technique, days_aggregate=22, m1zero=True, ret_nc_mom=True).to_numpy())
        realized_cumulants.append(rCumulants(process_df[column], method=technique, months_overlap=6).to_numpy())
    mvsek = np.squeeze(np.array(mvsek))
    realized_cumulants = np.squeeze(np.array(realized_cumulants))
    mvsek = pd.DataFrame(mvsek).T # each column is a path and each row is a moment (mean, variance, 3rd moment, 4th moment, skewness, excess kurtosis)
    realized_cumulants = pd.DataFrame(realized_cumulants).T
    mvsek = mvsek.mean(axis=1) # rowwise means
    realized_cumulants = realized_cumulants.mean(axis=1)
    
    # insert into database
    c.execute('INSERT INTO simulations (mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, start_date, end_date, interval, max_number_of_same_prices, NP_rc1, NP_rc2, NP_rc3, NP_rc4, NP_rm1, NP_rm2, NP_rm3, NP_rm4, NP_rskewness, NP_rexcess_kurtosis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, start_date, end_date, 'day', max_number_of_same_prices, realized_cumulants[0], realized_cumulants[1], realized_cumulants[2], realized_cumulants[3], mvsek[0], mvsek[1], mvsek[2], mvsek[3], mvsek[4], mvsek[5]))
    conn.commit()

def worker_function(params):
    conn = sqlite3.connect('simulation_database.db')
    create_simulation_and_save_it(conn, *params)
    conn.close()
    
# for v0 in tqdm(v0s):
#     for kappa in tqdm(kappas, leave=False):
#         for theta in tqdm(thetas, leave=False):
#             for sigma in tqdm(sigmas, leave=False):
#                 for mu in tqdm(mus, leave=False):
#                     for rho in tqdm(rhos, leave=False):
#                         create_simulation_and_save_it(start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin)

if __name__ == '__main__':
    parameter_list = [
        (start_date, end_date, time_points, T, S0, paths, v0, kappa, theta, sigma, mu, rho, burnin)
        for v0 in v0s
        for kappa in kappas
        for theta in thetas
        for sigma in sigmas
        for mu in mus
        for rho in rhos
    ]

    print('#Simulations:', len(parameter_list))

    with ProcessPoolExecutor() as executor:
        executor.map(worker_function, parameter_list)
        
# 47 Minuten laufzeit