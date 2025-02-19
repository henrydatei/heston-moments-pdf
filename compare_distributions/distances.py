import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

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

def Andersen_Darling_test(x_1, cdf_1, x_2, cdf_2):
    samples_1 = sample_from_cdf(x_1, cdf_1, 1000)
    samples_2 = sample_from_cdf(x_2, cdf_2, 1000)

    res = stats.anderson_ksamp([samples_1, samples_2])
    return res.statistic, res.pvalue