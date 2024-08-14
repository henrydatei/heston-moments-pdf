import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

def cornish_fisher_expansion(x, mean, var, skew, kurt):
    """Returns the Cornish-Fisher expansion of a random variable x with skewness skew and kurtosis kurt.
    Source: "Option Pricing Under Skewness and Kurtosis Using a Cornish-Fisher Expansion"

    Args:
        x (float): The value of the random variable.
        mean (float): The mean of the random variable.
        var (float): The variance of the random variable.
        skew (float): The skewness of the random variable.
        kurt (float): The kurtosis of the random variable.
    """
    x = (x - mean) / np.sqrt(var)
    a = (-skew)/(kurt/8 - (skew**2)/3)
    b = kurt/24 - (skew**2)/18
    p = (1 - kurt/8 + 5*(skew**2)/36)/(kurt/24 - (skew**2)/18) - 1/3 * ((skew**2)/36)/((kurt/24 - (skew**2)/18)**2)
    q = (-skew)/(kurt/8 - (skew**2)/3) - 1/18 * (skew * (1 - kurt/8 + 5*skew**2/36))/((kurt/24 - skew**2/18)**2) - 2/27 * (skew**3/216)/((kurt/24 - skew**2/18)**3)
    z = a/3 + np.cbrt((-q + x/b + np.sqrt((q-x/b)**2 + 4/27*p**3))/2) + np.cbrt((-q + x/b - np.sqrt((q-x/b)**2 + 4/27*p**3))/2)
    # print(f'{a=}')
    # print(f'{b=}')
    # print(f'{p=}')
    # print(f'{q=}')
    # print(f'{z=}')
    # print((q-x/b)**2 + 4/27*p**3)
    return 1/np.sqrt(2*np.pi) * np.exp(-z**2/2)/(z**2 * (kurt/8 - skew**2/6) + z*skew/3 + 1 - kurt/8 + 5*skew**2/36)

def linear_interpolate(x, x1, x2, y1, y2):
    """Linearly interpolates the value of y at x given the points (x1, y1) and (x2, y2).

    Args:
        x (float): The x value to interpolate at.
        x1 (float): The x value of the first point.
        x2 (float): The x value of the second point.
        y1 (float): The y value of the first point.
        y2 (float): The y value of the second point.
    """
    return y1 + (y2 - y1) / (x2 - x1) * (x - x1)

def find_s_k_from_skewness_exkurt_table(skewness, excess_kurtosis):
    actual_skewness_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2]
    actual_exkurt_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    s_values = [
        [0.000],
        [0.000,0.091,0.183,0.278,0.377,0.484],
        [0.000,0.084,0.169,0.255,0.345,0.439,0.540,0.652],
        [0.000,0.079,0.158,0.239,0.322,0.409,0.500,0.599,0.707],
        [0.000,0.075,0.150,0.227,0.306,0.387,0.472,0.561,0.658,0.764,0.885],
        [0.000,0.072,0.144,0.218,0.293,0.370,0.450,0.533,0.622,0.716,0.823],
        [0.000,0.069,0.139,0.210,0.282,0.357,0.431,0.510,0.593,0.680,0.776,1.000],
        [0.000,0.067,0.135,0.204,0.273,0.345,0.417,0.492,0.570,0.652,0.740,0.940],
        [0.000,0.065,0.131,0.198,0.265,0.334,0.404,0.476,0.551,0.629,0.712,0.895],
        [0.000,0.063,0.128,0.193,0.258,0.325,0.393,0.463,0.535,0.610,0.687,0.858],
        [0.000,0.062,0.125,0.189,0.252,0.317,0.384,0.451,0.520,0.593,0.666,0.828],
        [0.000,0.060,0.121,0.182,0.242,0.304,0.367,0.431,0.496,0.564,0.633,0.781,0.947],
        [0.000,0.058,0.116,0.175,0.234,0.293,0.354,0.415,0.478,0.542,0.607,0.746,0.897],
        [0.000,0.056,0.113,0.170,0.227,0.285,0.343,0.402,0.463,0.524,0.586,0.717,0.857,1.013],
        [0.000,0.055,0.110,0.165,0.221,0.277,0.334,0.391,0.450,0.509,0.569,0.694,0.826,0.971],
        [0.000,0.054,0.108,0.162,0.216,0.271,0.326,0.382,0.438,0.495,0.554,0.674,0.800,0.936,1.086],
        [0.000,0.049,0.099,0.148,0.198,0.248,0.298,0.349,0.400,0.451,0.502,0.608,0.717,0.830,0.948,1.074],
        [0.000,0.047,0.094,0.140,0.188,0.235,0.282,0.329,0.377,0.425,0.473,0.571,0.671,0.773,0.878,0.988,1.102],
        [0.000,0.046,0.090,0.135,0.181,0.226,0.271,0.317,0.363,0.409,0.455,0.548,0.643,0.739,0.837,0.937,1.041],
        [0.000,0.044,0.088,0.132,0.177,0.221,0.265,0.309,0.354,0.399,0.443,0.533,0.625,0.717,0.810,0.906,1.003]
    ]
    k_values = [
        [0.000],
        [0.426,0.428,0.432,0.439,0.452,0.470],
        [0.754,0.757,0.763,0.773,0.788,0.810,0.841,0.886],
        [1.026,1.028,1.035,1.046,1.063,1.086,1.117,1.159,1.218],
        [1.259,1.262,1.269,1.281,1.298,1.321,1.352,1.392,1.446,1.517,1.618],
        [1.466,1.469,1.476,1.488,1.505,1.529,1.559,1.598,1.648,1.712,1.796],
        [1.653,1.655,1.662,1.674,1.692,1.715,1.745,1.783,1.830,1.888,1.964,2.194],
        [1.823,1.826,1.833,1.845,1.862,1.885,1.915,1.951,1.996,2.051,2.121,2.319],
        [1.982,1.984,1.991,2.003,2.020,2.043,2.071,2.107,2.150,2.203,2.268,2.446],
        [2.129,2.132,2.139,2.150,2.167,2.190,2.218,2.252,2.295,2.345,2.406,2.569],
        [2.268,2.271,2.278,2.289,2.306,2.328,2.355,2.389,2.430,2.479,2.536,2.688],
        [2.525,2.528,2.534,2.546,2.562,2.583,2.610,2.642,2.680,2.727,2.780,2.916,3.109],
        [2.760,2.762,2.769,2.780,2.796,2.817,2.842,2.873,2.910,2.953,3.003,3.130,3.300],
        [2.978,2.980,2.986,2.997,3.013,3.033,3.058,3.087,3.123,3.164,3.212,3.330,3.486,3.696],
        [3.182,3.184,3.190,3.201,3.216,3.235,3.259,3.288,3.323,3.362,3.408,3.520,3.665,3.855],
        [3.374,3.376,3.383,3.393,3.408,3.427,3.450,3.478,3.511,3.550,3.593,3.701,3.837,4.012,4.243],
        [4.225,4.226,4.232,4.241,4.255,4.272,4.293,4.318,4.347,4.381,4.419,4.510,4.622,4.759,4.926,5.131],
        [4.962,4.964,4.970,4.978,4.991,5.007,5.026,5.050,5.077,5.107,5.143,5.225,5.326,5.446,5.588,5.756,5.954],
        [5.643,5.645,5.650,5.658,5.670,5.685,5.704,5.726,5.752,5.781,5.814,5.892,5.985,6.095,6.224,6.374,6.548],
        [6.295,6.297,6.302,6.310,6.321,6.336,6.354,6.375,6.400,6.428,6.460,6.534,6.623,6.728,6.849,6.989,7.148]
    ]
    if skewness in actual_skewness_values and excess_kurtosis in actual_exkurt_values:
        s_index = actual_skewness_values.index(skewness)
        k_index = actual_exkurt_values.index(excess_kurtosis)
        return s_values[k_index][s_index], k_values[k_index][s_index]
    if skewness < min(actual_skewness_values) and excess_kurtosis < min(actual_exkurt_values):
        return s_values[0][0], k_values[0][0]
    if skewness > max(actual_skewness_values) and excess_kurtosis > max(actual_exkurt_values):
        return s_values[-1][-1], k_values[-1][-1]
    if skewness < min(actual_skewness_values) and excess_kurtosis > max(actual_exkurt_values):
        return s_values[-1][0], k_values[-1][0]
    if skewness > max(actual_skewness_values) and excess_kurtosis < min(actual_exkurt_values):
        return s_values[0][-1], k_values[0][-1]
    s_index = np.searchsorted(actual_skewness_values, skewness) # bigger element
    k_index = np.searchsorted(actual_exkurt_values, excess_kurtosis) # bigger element
    # ... (domain of validity?)


# print(cornish_fisher_expansion(0.5, 0.5, 2))

# Calculate skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_exkurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_exkurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_exkurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_skew, normal_exkurt)
print(lognorm_skew, lognorm_exkurt)
print(t_skew, t_exkurt)
print(nct_skew, nct_exkurt)

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Cornish-Fisher expansion
normal_expansion = cornish_fisher_expansion(x, normal_mean, normal_var, normal_skew, normal_exkurt)
lognorm_expansion = cornish_fisher_expansion(x, lognorm_mean, lognorm_var,lognorm_skew, lognorm_exkurt)
t_expansion = cornish_fisher_expansion(x, t_mean, t_var, t_skew, t_exkurt)
nct_expansion = cornish_fisher_expansion(x, nct_mean, nct_var, nct_skew, nct_exkurt)

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Normal Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Log-Normal Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('t Distribution and Cornish-Fisher Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Cornish-Fisher Expansion')
plt.title('Non-central t Distribution and Cornish-Fisher Expansion')
plt.legend()

plt.tight_layout()
plt.show()
