import sqlite3
from tqdm import tqdm

conn = sqlite3.connect('simulations.db')
c = conn.cursor()

cols = [
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

for col in tqdm(cols):
    c.execute(f"CREATE INDEX idx_{col} ON simulations({col} ASC)")
    conn.commit()