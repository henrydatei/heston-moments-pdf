import sqlite3
import os
import sys
import pandas as pd
from tqdm import tqdm
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_database.database_utils import add_column, update_value

db_file = 'simulations.db'

def create_table():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mu REAL,
        kappa REAL,
        theta REAL,
        sigma REAL,
        rho REAL,
        v0 REAL,
        time_points INTEGER,
        burnin INTEGER,
        T INTEGER,
        S0 REAL,
        paths INTEGER,
        start_date TEXT,
        end_date TEXT,
        interval TEXT,
        max_number_of_same_prices INTEGER,
        NP_rc1 REAL,
        NP_rc2 REAL,
        NP_rc3 REAL,
        NP_rc4 REAL,
        NP_rm1 REAL,
        NP_rm2 REAL,
        NP_rm3 REAL,
        NP_rm4 REAL,
        NP_rskewness REAL,
        NP_rexcess_kurtosis REAL,
        seed INTEGER,
        UNIQUE(v0, kappa, theta, sigma, mu, rho, time_points, burnin, T, S0, paths, start_date, end_date, seed)
    )
    ''')
    conn.commit()
    conn.close()

def insert_csv_to_db(csv_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    df = pd.read_csv(csv_file, header=0, names=[
        'mu', 'kappa', 'theta', 'sigma', 'rho', 'v0', 'time_points', 'burnin', 'T', 'S0', 'paths', 
        'start_date', 'end_date', 'interval', 'max_number_of_same_prices', 'NP_rc1', 'NP_rc2', 'NP_rc3', 'NP_rc4', 
        'NP_rm1', 'NP_rm2', 'NP_rm3', 'NP_rm4', 'NP_rskewness', 'NP_rexcess_kurtosis', 'seed'
    ])

    for _, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT INTO simulations (
                    mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, 
                    start_date, end_date, interval, max_number_of_same_prices, NP_rc1, NP_rc2, NP_rc3, NP_rc4, 
                    NP_rm1, NP_rm2, NP_rm3, NP_rm4, NP_rskewness, NP_rexcess_kurtosis, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(row))
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    
def parse_log_file(log_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - Worker \d+ result: )')
    
    # get total lines in log_file
    total_lines = 0
    with open(log_file, 'r') as file:
        for _ in file:
            total_lines += 1
    
    with open(log_file, 'r') as file:
        for line in tqdm(file, total=total_lines, desc='Processing log file'):
            if pattern.match(line):
                line = pattern.sub('', line)
                line = re.sub(r'\(|\)', '', line)
                line = re.sub(r'np\.float64|np\.int64', '', line)
                values = line.strip().split(', ')
                try:
                    values = [float(v) if '.' in v or 'e' in v.lower() else int(v) if v.isdigit() else v.strip('"') for v in values]
                    cursor.execute('''
                        INSERT INTO simulations (
                            mu, kappa, theta, sigma, rho, v0, time_points, burnin, T, S0, paths, 
                            start_date, end_date, interval, max_number_of_same_prices, NP_rc1, NP_rc2, NP_rc3, NP_rc4, 
                            NP_rm1, NP_rm2, NP_rm3, NP_rm4, NP_rskewness, NP_rexcess_kurtosis, seed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', values)
                except (sqlite3.IntegrityError):
                    pass
                except ValueError as e:
                    print(f'Error {e}: {line}')
    conn.commit()
    conn.close()

def process_results_folder():
    base_folder = '/Users/henryhaustein/Downloads/heston-moments-pdf/simulation_database/results'
    create_table()
    subfolders = sorted(os.listdir(base_folder))
    print(f'Found {len(subfolders)} subjobs')
    
    for subfolder in tqdm(subfolders, desc='Processing subjobs'):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
            for csv_file in tqdm(csv_files, desc=f'Processing {subfolder}', leave=False):
                csv_path = os.path.join(subfolder_path, csv_file)
                insert_csv_to_db(csv_path)
                
def check_feller_condition():
    add_column('simulations', 'feller_condition', 'INTEGER')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM simulations')
    simulations = cursor.fetchall()
    for simulation in tqdm(simulations, desc='Checking Feller Condition'):
        mu, kappa, theta, sigma, rho, v0 = simulation[1:7]
        if 2 * kappa * theta > sigma ** 2:
            update_value('simulations', 'feller_condition', 1, mu, kappa, theta, sigma, rho, v0)
        else:
            update_value('simulations', 'feller_condition', 0, mu, kappa, theta, sigma, rho, v0)

def round_numbers():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('UPDATE simulations SET mu=ROUND(mu, 6), kappa=ROUND(kappa, 6), theta=ROUND(theta, 6), sigma=ROUND(sigma, 6), rho=ROUND(rho, 6), v0=ROUND(v0, 6)')
    conn.commit()

def calc_theoretical_mean():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # T = time interval to consider a return, here 1/12
    # mean = (mu - theta/2)*T
    add_column('simulations', 'theoretical_mean', 'REAL')
    cursor.execute('UPDATE simulations SET theoretical_mean=(mu - theta/2)*1/12')
    conn.commit()
    
def calc_theoretical_variance():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # T = time interval to consider a return, here 1/12
    # sigma2 = sigma ** 2
    # kappa2 = kappa ** 2
    # kappa3 = kappa ** 3
    # numerator = (sigma * (- 4 * kappa * rho + sigma) * theta + np.exp(kappa * T) * (- (sigma2 * theta) - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * (4 * rho + sigma * T) * theta + kappa3 * T * (4 * theta + T * (-2 * mu + theta)**2)))
    # denominator = (4 * np.exp(kappa * T) * kappa3)
    # variance = numerator / denominator
    add_column('simulations', 'theoretical_variance', 'REAL')
    cursor.execute('''
        UPDATE simulations SET theoretical_variance = 
        (
            sigma * (
                - 4 * kappa * rho + sigma
            ) * theta + 
            exp(kappa * 1/12) * (
                - (sigma * sigma * theta) - 4 * kappa * kappa * rho * sigma * 1/12 * theta + kappa * sigma * (
                    4 * rho + sigma * 1/12
                ) * theta + kappa * kappa * kappa * 1/12 * (
                    4 * theta + 1/12 * (-2 * mu + theta) * (-2 * mu + theta)
                )
            )
        ) /
        (
            4 * exp(kappa * 1/12) * kappa * kappa * kappa
        )
    ''')
    conn.commit()

def calc_theoretical_skewness():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # T = time interval to consider a return, here 1/12
    # sigma2 = sigma ** 2
    # kappa2 = kappa ** 2
    # kappa3 = kappa ** 3
    # kappa5 = kappa ** 5
    # numerator = -3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + kappa * sigma * (8 * rho - sigma * T) + 4 * kappa2 * (-1 + rho * sigma * T)) * theta - np.exp(kappa * T) * (-3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma2 + 4 * kappa3 * T + kappa * sigma * (8 * rho + sigma * T) - 4 * kappa2 * (1 + rho * sigma * T)) * theta)
    # denominator = (np.exp(kappa * T) * kappa5 * (((-1 + np.exp(-kappa * T)) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * ((4 - 4 / np.exp(kappa * T)) * rho + sigma * T) * theta + kappa3 * T * 4 * theta) / kappa3) ** 1.5)
    # skewness = numerator / denominator
    add_column('simulations', 'theoretical_skewness', 'REAL')
    cursor.execute('''
        UPDATE simulations SET theoretical_skewness = (
            -3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma * sigma + kappa * sigma * (8 * rho - sigma * 1/12) + 4 * kappa * kappa * (-1 + rho * sigma * 1/12)) * theta - exp(kappa * 1/12) * (-3 * (2 * kappa * rho - sigma) * sigma * (-2 * sigma * sigma + 4 * kappa * kappa * kappa * 1/12 + kappa * sigma * (8 * rho + sigma * 1/12) - 4 * kappa * kappa * (1 + rho * sigma * 1/12)) * theta)
            ) / (
            exp(kappa * 1/12) * kappa * kappa * kappa * kappa * kappa * sqrt((((-1 + exp(-kappa * 1/12)) * sigma * sigma * theta - 4 * kappa * kappa * rho * sigma * 1/12 * theta + kappa * sigma * ((4 - 4 / exp(kappa * 1/12)) * rho + sigma * 1/12) * theta + kappa * kappa * kappa * 1/12 * 4 * theta) / (kappa * kappa * kappa)) * (((-1 + exp(-kappa * 1/12)) * sigma * sigma * theta - 4 * kappa * kappa * rho * sigma * 1/12 * theta + kappa * sigma * ((4 - 4 / exp(kappa * 1/12)) * rho + sigma * 1/12) * theta + kappa * kappa * kappa * 1/12 * 4 * theta) / (kappa * kappa * kappa)) * (((-1 + exp(-kappa * 1/12)) * sigma * sigma * theta - 4 * kappa * kappa * rho * sigma * 1/12 * theta + kappa * sigma * ((4 - 4 / exp(kappa * 1/12)) * rho + sigma * 1/12) * theta + kappa * kappa * kappa * 1/12 * 4 * theta) / (kappa * kappa * kappa)))
        )
    ''')
    conn.commit()
    
def calc_theoretical_kurtosis():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # T = time interval to consider a return, here 1/12
    # sigma2 = sigma**2  
    # kappa2 = kappa**2 
    # kappa3 = kappa**3
    # kappa4 = kappa**4
    # kappa7 = kappa**7
    # exp_kappa_t = np.exp(kappa * T)
    # theta_lambdaj_vj = 4 * theta
    # kurtosis = (3 * sigma2 * (- 4 * kappa * rho + sigma)**2 * theta * (sigma2 + 2 * kappa * theta) + 12 * exp_kappa_t * sigma * theta * (7 * sigma**5 - kappa * sigma**3 * (56 * rho*sigma - 5 * sigma2 * T + theta) + kappa2 * sigma2 * (- 40 * rho * sigma2 * T + sigma**3 * T**2 + 8 * rho * theta + sigma * (24 + 136 * rho**2 + T * theta)) - 4 * kappa3 * sigma * (24 * rho**3 * sigma - 3 * sigma2 * T + 4 * rho**2 * (- 6 * sigma2 * T + theta) + 2 * rho * sigma * (12 + sigma2 * T**2 + T * theta)) - 4 * kappa**5 * rho * T * (- 8 * rho * sigma + 4 * rho**2 * sigma2 * T + theta_lambdaj_vj) + kappa4 * sigma * (8 - 48 * rho * sigma * T - 64 * rho**3 * sigma * T + 4 * rho**2 * (16 + 5 * sigma2 * T**2 + 4 * T * theta) + T * theta_lambdaj_vj)) + exp_kappa_t**2 * (- 87 * sigma**6 * theta + 6 * kappa * sigma**4 * theta * (116 * rho * sigma + 5 * sigma2 * T + theta) + 6 * kappa3 * sigma2 * theta * (192 * rho**3 * sigma + 16 * rho**2 * (6 * sigma2 * T + theta) + 16 * rho * sigma * (12 + T * theta) + sigma2 * T * (24 + T*theta)) - 12 * kappa2 * sigma**3 * theta * (20 * rho * sigma2 * T + 4 * rho * theta + sigma * (24 + 140 * rho**2 + T * theta)) - 48 * kappa**6 * rho * sigma * T**2 * theta * theta_lambdaj_vj - 12 * kappa4 * sigma2 * theta * (8 + 32 * rho**3 * sigma * T + 16 * rho**2 * (4 + T * theta) + 4 * rho * sigma * T * (12 + T * theta) + T * theta_lambdaj_vj) + 2 * kappa7 * T * (3 * T * theta_lambdaj_vj**2) + 12 * kappa**5 * sigma * T * theta * (4 * rho * theta_lambdaj_vj + sigma * (8 + 8 * rho**2 * (4 + T * theta) + T * theta_lambdaj_vj)))) / (2 * exp_kappa_t**2 * kappa7 * (((- 1 + 1 / exp_kappa_t) * sigma2 * theta - 4 * kappa2 * rho * sigma * T * theta + kappa * sigma * ((4 - 4 / exp_kappa_t) * rho + sigma * T) * theta + kappa3 * T * theta_lambdaj_vj) / kappa3)**2)
    add_column('simulations', 'theoretical_kurtosis', 'REAL')
    cursor.execute('''     
        UPDATE simulations SET theoretical_kurtosis = (
            (3 * sigma*sigma * (- 4 * kappa * rho + sigma)*(- 4 * kappa * rho + sigma) * theta * (sigma*sigma + 2 * kappa * theta) + 12 * exp(kappa * 1/12) * sigma * theta * (7 * sigma*sigma*sigma*sigma*sigma - kappa * sigma*sigma*sigma * (56 * rho*sigma - 5 * sigma*sigma * 1/12 + theta) + kappa*kappa * sigma*sigma * (- 40 * rho * sigma*sigma * 1/12 + sigma*sigma*sigma * 1/12*1/12 + 8 * rho * theta + sigma * (24 + 136 * rho*rho + 1/12 * theta)) - 4 * kappa*kappa*kappa * sigma * (24 * rho*rho*rho * sigma - 3 * sigma*sigma * 1/12 + 4 * rho*rho * (- 6 * sigma*sigma * 1/12 + theta) + 2 * rho * sigma * (12 + sigma*sigma * 1/12*1/12 + 1/12 * theta)) - 4 * kappa*kappa*kappa*kappa*kappa * rho * 1/12 * (- 8 * rho * sigma + 4 * rho*rho * sigma*sigma * 1/12 + 4 * theta) + kappa*kappa*kappa*kappa * sigma * (8 - 48 * rho * sigma * 1/12 - 64 * rho*rho*rho * sigma * 1/12 + 4 * rho*rho * (16 + 5 * sigma*sigma * 1/12*1/12 + 4 * 1/12 * theta) + 1/12 * 4 * theta)) + exp(kappa * 1/12) * exp(kappa * 1/12) * (- 87 * sigma*sigma*sigma*sigma*sigma*sigma * theta + 6 * kappa * sigma*sigma*sigma*sigma * theta * (116 * rho * sigma + 5 * sigma*sigma * 1/12 + theta) + 6 * kappa*kappa*kappa * sigma*sigma * theta * (192 * rho*rho*rho * sigma + 16 * rho*rho * (6 * sigma*sigma * 1/12 + theta) + 16 * rho * sigma * (12 + 1/12 * theta) + sigma*sigma * 1/12 * (24 + 1/12*theta)) - 12 * kappa*kappa * sigma*sigma*sigma * theta * (20 * rho * sigma*sigma * 1/12 + 4 * rho * theta + sigma * (24 + 140 * rho*rho + 1/12 * theta)) - 48 * kappa*kappa*kappa*kappa*kappa*kappa * rho * sigma * 1/12*1/12 * theta * 4 * theta - 12 * kappa*kappa*kappa*kappa * sigma*sigma * theta * (8 + 32 * rho*rho*rho * sigma * 1/12 + 16 * rho*rho * (4 + 1/12 * theta) + 4 * rho * sigma * 1/12 * (12 + 1/12 * theta) + 1/12 * 4 * theta) + 2 * kappa*kappa*kappa*kappa*kappa*kappa*kappa * 1/12 * (3 * 1/12 * 4 * theta * 4 * theta) + 12 * kappa*kappa*kappa*kappa*kappa * sigma * 1/12 * theta * (4 * rho * 4 * theta + sigma * (8 + 8 * rho*rho * (4 + 1/12 * theta) + 1/12 * 4 * theta)))) / (2 * exp(kappa * 1/12) * exp(kappa * 1/12) * kappa*kappa*kappa*kappa*kappa*kappa*kappa * (((- 1 + 1 / (exp(kappa * 1/12))) * sigma*sigma * theta - 4 * kappa*kappa * rho * sigma * 1/12 * theta + kappa * sigma * ((4 - 4 / (exp(kappa * 1/12))) * rho + sigma * 1/12) * theta + kappa*kappa*kappa * 1/12 * 4 * theta) / (kappa * kappa * kappa))*(((- 1 + 1 / (exp(kappa * 1/12))) * sigma*sigma * theta - 4 * kappa*kappa * rho * sigma * 1/12 * theta + kappa * sigma * ((4 - 4 / (exp(kappa * 1/12))) * rho + sigma * 1/12) * theta + kappa*kappa*kappa * 1/12 * 4 * theta) / (kappa * kappa * kappa)))
        )
    ''')
    conn.commit()

if __name__ == "__main__":
    # process_results_folder()
    # parse_log_file('/Users/henryhaustein/Downloads/heston-moments-pdf/simulation_database/results/log.txt')
    # round_numbers()
    # check_feller_condition()
    calc_theoretical_mean()
    calc_theoretical_variance()
    calc_theoretical_skewness()
    calc_theoretical_kurtosis()
