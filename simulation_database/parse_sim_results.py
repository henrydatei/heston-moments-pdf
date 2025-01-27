import sqlite3
import os
import pandas as pd
from tqdm import tqdm
import re

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
    with open(log_file, 'r') as file:
        for line in file:
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

if __name__ == "__main__":
    # process_results_folder()
    parse_log_file('/Users/henryhaustein/Downloads/heston-moments-pdf/simulation_database/results/log.txt')
