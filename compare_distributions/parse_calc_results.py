import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_database.database_utils import update_multiple_values, add_column

def update_csv_to_db(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: Skipping file {csv_file}")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    result_columns = [
        'GC_cum_KS_stat', 'GC_cum_KS_p', 'GC_cum_CV_stat', 'GC_cum_CV_p', 'GC_cum_AD_stat', 'GC_cum_AD_p',
        'GC_mom_KS_stat', 'GC_mom_KS_p', 'GC_mom_CV_stat', 'GC_mom_CV_p', 'GC_mom_AD_stat', 'GC_mom_AD_p',
        'GC_pos_cum_KS_stat', 'GC_pos_cum_KS_p', 'GC_pos_cum_CV_stat', 'GC_pos_cum_CV_p', 'GC_pos_cum_AD_stat', 'GC_pos_cum_AD_p',
        'GC_pos_mom_KS_stat', 'GC_pos_mom_KS_p', 'GC_pos_mom_CV_stat', 'GC_pos_mom_CV_p','GC_pos_mom_AD_stat', 'GC_pos_mom_AD_p',
        'EW_cum_KS_stat', 'EW_cum_KS_p', 'EW_cum_CV_stat', 'EW_cum_CV_p', 'EW_cum_AD_stat', 'EW_cum_AD_p',
        'EW_mom_KS_stat', 'EW_mom_KS_p', 'EW_mom_CV_stat', 'EW_mom_CV_p', 'EW_mom_AD_stat', 'EW_mom_AD_p',
        'EW_pos_cum_KS_stat', 'EW_pos_cum_KS_p', 'EW_pos_cum_CV_stat', 'EW_pos_cum_CV_p', 'EW_pos_cum_AD_stat', 'EW_pos_cum_AD_p',
        'EW_pos_mom_KS_stat', 'EW_pos_mom_KS_p', 'EW_pos_mom_CV_stat', 'EW_pos_mom_CV_p', 'EW_pos_mom_AD_stat', 'EW_pos_mom_AD_p',
        'CF_cum_KS_stat', 'CF_cum_KS_p', 'CF_cum_CV_stat', 'CF_cum_CV_p', 'CF_cum_AD_stat', 'CF_cum_AD_p',
        'CF_mom_KS_stat', 'CF_mom_KS_p', 'CF_mom_CV_stat', 'CF_mom_CV_p', 'CF_mom_AD_stat', 'CF_mom_AD_p',
        'SP_cum_KS_stat', 'SP_cum_KS_p', 'SP_cum_CV_stat', 'SP_cum_CV_p', 'SP_cum_AD_stat', 'SP_cum_AD_p',
        'SP_mom_KS_stat', 'SP_mom_KS_p', 'SP_mom_CV_stat', 'SP_mom_CV_p', 'SP_mom_AD_stat', 'SP_mom_AD_p'
    ]
    
    for _, row in df.iterrows():
        values = {}
        for col in result_columns:
            # Es kann vorkommen, dass in manchen Zeilen ein Ergebnis nicht gesetzt ist.
            # Daher verwenden wir get(), um im Fehlerfall None zu erhalten.
            values[col] = row.get(col)
        
        mu = row.get('mu')
        kappa = row.get('kappa')
        theta = row.get('theta')
        sigma = row.get('sigma')
        rho = row.get('rho')
        v0 = row.get('v0')
        
        # Aktualisiere den Datensatz in der Datenbank.
        update_multiple_values('simulations', values, mu, kappa, theta, sigma, rho, v0)

def process_results_folder():
    base_folder = '/Users/henryhaustein/Downloads/heston-moments-pdf/compare_distributions/results'
    
    add_column('simulations', 'GC_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'GC_cum_KS_p', 'FLOAT')
    add_column('simulations', 'GC_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'GC_cum_CV_p', 'FLOAT')
    add_column('simulations', 'GC_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'GC_cum_AD_p', 'FLOAT')

    add_column('simulations', 'GC_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'GC_mom_KS_p', 'FLOAT')
    add_column('simulations', 'GC_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'GC_mom_CV_p', 'FLOAT')
    add_column('simulations', 'GC_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'GC_mom_AD_p', 'FLOAT')

    add_column('simulations', 'GC_pos_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_cum_KS_p', 'FLOAT')
    add_column('simulations', 'GC_pos_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_cum_CV_p', 'FLOAT')
    add_column('simulations', 'GC_pos_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_cum_AD_p', 'FLOAT')

    add_column('simulations', 'GC_pos_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_mom_KS_p', 'FLOAT')
    add_column('simulations', 'GC_pos_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_mom_CV_p', 'FLOAT')
    add_column('simulations', 'GC_pos_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'GC_pos_mom_AD_p', 'FLOAT')

    add_column('simulations', 'EW_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'EW_cum_KS_p', 'FLOAT')
    add_column('simulations', 'EW_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'EW_cum_CV_p', 'FLOAT')
    add_column('simulations', 'EW_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'EW_cum_AD_p', 'FLOAT')

    add_column('simulations', 'EW_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'EW_mom_KS_p', 'FLOAT')
    add_column('simulations', 'EW_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'EW_mom_CV_p', 'FLOAT')
    add_column('simulations', 'EW_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'EW_mom_AD_p', 'FLOAT')

    add_column('simulations', 'EW_pos_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_cum_KS_p', 'FLOAT')
    add_column('simulations', 'EW_pos_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_cum_CV_p', 'FLOAT')
    add_column('simulations', 'EW_pos_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_cum_AD_p', 'FLOAT')

    add_column('simulations', 'EW_pos_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_mom_KS_p', 'FLOAT')
    add_column('simulations', 'EW_pos_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_mom_CV_p', 'FLOAT')
    add_column('simulations', 'EW_pos_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'EW_pos_mom_AD_p', 'FLOAT')

    add_column('simulations', 'CF_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'CF_cum_KS_p', 'FLOAT')
    add_column('simulations', 'CF_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'CF_cum_CV_p', 'FLOAT')
    add_column('simulations', 'CF_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'CF_cum_AD_p', 'FLOAT')

    add_column('simulations', 'CF_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'CF_mom_KS_p', 'FLOAT')
    add_column('simulations', 'CF_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'CF_mom_CV_p', 'FLOAT')
    add_column('simulations', 'CF_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'CF_mom_AD_p', 'FLOAT')

    add_column('simulations', 'SP_cum_KS_stat', 'FLOAT')
    add_column('simulations', 'SP_cum_KS_p', 'FLOAT')
    add_column('simulations', 'SP_cum_CV_stat', 'FLOAT')
    add_column('simulations', 'SP_cum_CV_p', 'FLOAT')
    add_column('simulations', 'SP_cum_AD_stat', 'FLOAT')
    add_column('simulations', 'SP_cum_AD_p', 'FLOAT')

    add_column('simulations', 'SP_mom_KS_stat', 'FLOAT')
    add_column('simulations', 'SP_mom_KS_p', 'FLOAT')
    add_column('simulations', 'SP_mom_CV_stat', 'FLOAT')
    add_column('simulations', 'SP_mom_CV_p', 'FLOAT')
    add_column('simulations', 'SP_mom_AD_stat', 'FLOAT')
    add_column('simulations', 'SP_mom_AD_p', 'FLOAT')
    
    subfolders = sorted(os.listdir(base_folder))
    print(f'Found {len(subfolders)} subjobs')
    
    for subfolder in tqdm(subfolders, desc='Processing subjobs'):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
            for csv_file in tqdm(csv_files, desc=f'Processing {subfolder}', leave=False):
                csv_path = os.path.join(subfolder_path, csv_file)
                update_csv_to_db(csv_path)

if __name__ == "__main__":
    process_results_folder()
