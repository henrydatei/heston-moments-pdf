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

if __name__ == "__main__":
    # process_results_folder()
    # parse_log_file('/Users/henryhaustein/Downloads/heston-moments-pdf/simulation_database/results/log.txt')
    round_numbers()
    # check_feller_condition()
