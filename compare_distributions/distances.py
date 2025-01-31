from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expansion_methods.all_methods import scipy_mvsek_to_cumulants, gram_charlier_expansion
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate
from simulation_database.database_utils import add_column

from code_from_haozhe.GramCharlier_expansion import Expansion_GramCharlier

def pdf_to_cdf(x, pdf, normalize=True):
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf) * dx
    if normalize:
        cdf = cdf / cdf[-1]
    return cdf

def sample_from_pdf(x, pdf, n_samples):
    np.random.seed(0)
    return np.random.choice(x, size=n_samples, p=pdf/np.sum(pdf))

def sample_from_cdf(x, cdf, n_samples):
    np.random.seed(0)
    inv_cdf = interp1d(cdf, x, kind='linear', fill_value='extrapolate')
    uniform_samples = np.random.uniform(0, 1, n_samples)
    return inv_cdf(uniform_samples)

def KS_test_sample_from_pdf(x_1, pdf_1, x_2, pdf_2):
    samples_1 = sample_from_pdf(x_1, pdf_1, 1000)
    samples_2 = sample_from_pdf(x_2, pdf_2, 1000)

    ks_statistic, p_value = stats.ks_2samp(samples_1, samples_2)
    return ks_statistic, p_value

def KS_test_sample_from_cdf(x_1, cdf_1, x_2, cdf_2):
    samples_1 = sample_from_cdf(x_1, cdf_1, 1000)
    samples_2 = sample_from_cdf(x_2, cdf_2, 1000)

    ks_statistic, p_value = stats.ks_2samp(samples_1, samples_2)
    return ks_statistic, p_value

def Cramer_von_Mises_test(x_1, cdf_1, x_2, cdf_2):
    samples_1 = sample_from_cdf(x_1, cdf_1, 1000)
    samples_2 = sample_from_cdf(x_2, cdf_2, 1000)

    res = stats.cramervonmises_2samp(samples_1, samples_2)
    return res.statistic, res.pvalue