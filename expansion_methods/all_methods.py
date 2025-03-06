import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import cmath

i = 1j

# These are the roots of He_6(z), each root is symmetric about the real axis, np.real() is nessary to remove the imaginary part which comes from numerical error
Z_ROOT1 = np.real(cmath.sqrt(5 - (5**(2/3) * (1 + i * np.sqrt(3)) / (2 * (2 + i * np.sqrt(6)))**(1/3)) - ((1 - i * np.sqrt(3)) * (5 * (2 + i * np.sqrt(6)))**(1/3) / 2**(2/3))))

Z_ROOT2 = np.real(cmath.sqrt(5 - (5**(2/3) * (1 - i * np.sqrt(3)) / (2 * (2 + i * np.sqrt(6)))**(1/3)) - ((1 + i * np.sqrt(3)) * (5 * (2 + i * np.sqrt(6)))**(1/3) / 2**(2/3))))

Z_ROOT3 = np.real(cmath.sqrt(5 + (10**(2/3) / (2 + i * np.sqrt(6))**(1/3)) + ((10 * (2 + i * np.sqrt(6)))**(1/3))))

def scipy_mvsek_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

def moments_to_cumulants(first_moment, second_moment, third_moment, fourth_moment):
    first_cumulant = first_moment
    second_cumulant = second_moment - first_moment**2
    third_cumulant = third_moment - 3*first_moment*second_moment + 2*first_moment**3
    fourth_cumulant = fourth_moment - 4*first_moment*third_moment - 3*second_moment**2 + 12*first_moment**2*second_moment - 6*first_moment**4
    return first_cumulant, second_cumulant, third_cumulant, fourth_cumulant

def moments_to_mvsek(first_moment, second_moment, third_moment, fourth_moment):
    mean = first_moment
    variance = second_moment - first_moment**2
    skewness = (third_moment - 3*first_moment*second_moment + 2*first_moment**3) / variance**1.5
    excess_kurtosis = (fourth_moment - 4*first_moment*third_moment + 6*first_moment**2*second_moment - 3*first_moment**4) / variance**2 - 3
    return mean, variance, skewness, excess_kurtosis

def cumulants_to_mvsek(first_cumulant, second_cumulant, third_cumulant, fourth_cumulant):
    mean = first_cumulant
    variance = second_cumulant
    skewness = third_cumulant / variance**1.5
    excess_kurtosis = fourth_cumulant / variance**2
    return mean, variance, skewness, excess_kurtosis

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)
    
def normal_expansion(x, mean, variance):
    return norm.pdf(x, loc = mean, scale = np.sqrt(variance))
    
def gram_charlier_expansion(x, mean, variance, third_cumulant, fourth_cumulant, fakasawa = False):
    z = (x - mean) / np.sqrt(variance)
    if fakasawa:
        return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/(6 * variance**1.5) * hermite_polynomial(3, z) + (fourth_cumulant / variance**2 - 3)/24 * hermite_polynomial(4, z))
    else:
        return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/(6 * variance**1.5) * hermite_polynomial(3, z) + (fourth_cumulant / variance**2)/24 * hermite_polynomial(4, z))

def edgeworth_expansion(x, mean, variance, third_cumulant, fourth_cumulant, fakasawa = False):
    z = (x - mean) / np.sqrt(variance)
    if fakasawa:
        return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/(6 * variance**1.5) * hermite_polynomial(3, z) + (fourth_cumulant / variance**2 - 3)/24 * hermite_polynomial(4, z) + (third_cumulant/(variance**1.5))**2/72 * hermite_polynomial(6, z))
    else:
        return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/(6 * variance**1.5) * hermite_polynomial(3, z) + fourth_cumulant/(24 * variance**2) * hermite_polynomial(4, z) + (third_cumulant/(variance**1.5))**2/72 * hermite_polynomial(6, z))

def intersection_lines(a,b,c,d):
    x = (d-b)/(a-c)
    y = (a*d - c*b)/(a-c)
    return x, y

def get_positivity_boundary_lines(z):
    """Returns the lines that define the positivity boundary of the GC density function in the form of skew = a*kurt + b and kurt = c*skew + d.

    Args:
        z (float): The value of z in the GC density function.
    """
    a = (z**4 - 6*z**2 + 3)/(12*z - 4*z**3)
    b = 24/(12*z - 4*z**3)
    c = (12*z - 4*z**3)/(z**4 - 6*z**2 + 3)
    d = -24/(z**4 - 6*z**2 + 3)
    return a, b, c, d

def linear_boundary_lines(z, k):
    a, b, _, _ = get_positivity_boundary_lines(z)
    return a*k + b

def get_intersections_gc(STEPS = 1000, Z_START = -10, Z_END = -np.sqrt(3)):
    all_z, stepsize = np.linspace(Z_START, Z_END, STEPS, retstep=True)
    intersections = [(4,0), (0,0)]

    for z in all_z:
        skew_a1, skew_b1, _, _ = get_positivity_boundary_lines(z)
        skew_a2, skew_b2, _, _ = get_positivity_boundary_lines(z+stepsize)
        inter_x, inter_y = intersection_lines(skew_a1, skew_b1, skew_a2, skew_b2)
        intersections.append((inter_x, inter_y))
        
    return intersections

def logistic_map(x, a,b):
    return a + (b-a)/(1+np.exp(-x))

def neg_log_likelihood_gc(params, data):
    mu, variance, s, k = params
    s, k = transform_skew_exkurt_into_positivity_region(s, k, get_intersections_gc())
    likelihoods = gram_charlier_expansion(data, mu, variance, s, k)
    return -np.sum(np.log(likelihoods))

def transform_skew_exkurt_into_positivity_region(skew, exkurt, intersections):
    skew_sign = np.sign(skew)
    skew = abs(skew)
    new_exkurt = logistic_map(exkurt, 0, 4)

    if new_exkurt == 4:
        return 0, 4
    
    # find i such that intersections[i][0] < new_kurt <= intersections[i+1][0]
    intersections.sort(key = lambda x: x[0])
    for i in range(len(intersections)-1):
        if intersections[i][0] < new_exkurt <= intersections[i+1][0]:
            break

    k_i, s_i = intersections[i]
    k_i2, s_i2 = intersections[i+1]
    a_i = (s_i * k_i2 - k_i * s_i2)/(k_i2 - k_i)
    b_i = (s_i2 - s_i)/(k_i2 - k_i)
    s_u = a_i + b_i * new_exkurt
    s_l = -s_u
    # print(i, k_i, s_i, k_i2, s_i2, a_i, b_i, s_u, s_l)

    new_skew = logistic_map(skew, s_l, s_u)
    new_skew = skew_sign * new_skew

    return new_skew, new_exkurt

def gram_charlier_expansion_positivity_constraint(x, mean, variance, skewness, exkurt, fakasawa = False):
    new_skew, new_exkurt = transform_skew_exkurt_into_positivity_region(skewness, exkurt, get_intersections_gc())
    expansion = gram_charlier_expansion(x, *scipy_mvsek_to_cumulants(mean, variance, new_skew, new_exkurt), fakasawa)
        
    return expansion

def parabolic_boundary_lines(z, k):
    s = np.sqrt(-72/hermite_polynomial(6,z) - 3*k*hermite_polynomial(4,z)/hermite_polynomial(6,z) + 36*hermite_polynomial(3,z)**2/hermite_polynomial(6,z)**2) - 6*hermite_polynomial(3,z)/hermite_polynomial(6,z)
    return s

def parabolic_boundary_lines_coef(z):
    a = -72/hermite_polynomial(6,z) + 36*hermite_polynomial(3,z)**2/hermite_polynomial(6,z)**2
    b = -3*hermite_polynomial(4,z)/hermite_polynomial(6,z)
    c = -6*hermite_polynomial(3,z)/hermite_polynomial(6,z)
    return a, b, c

def intersection_parabolas(a, b, c, d, e, f):
    '''
    Finds intersection of sqrt(a+b*k)+c = sqrt(d+e*k)+f
    '''
    k1 = (2*(b - e)**2*(-c + f)*np.sqrt(-a*b*e + a*e**2 + b**2*d + b*c**2*e - 2*b*c*e*f - b*d*e + b*e*f**2) + (b**2 - 2*b*e + e**2)*(-a*b + a*e + b*c**2 - 2*b*c*f + b*d + b*f**2 + c**2*e - 2*c*e*f - d*e + e*f**2))/((b - e)**2*(b**2 - 2*b*e + e**2))
    s1 = c + np.sqrt(a + b*(2*(b - e)**2*(-c + f)*np.sqrt(-a*b*e + a*e**2 + b**2*d + b*c**2*e - 2*b*c*e*f - b*d*e + b*e*f**2) + (b**2 - 2*b*e + e**2)*(-a*b + a*e + b*c**2 - 2*b*c*f + b*d + b*f**2 + c**2*e - 2*c*e*f - d*e + e*f**2))/((b - e)**2*(b**2 - 2*b*e + e**2)))
    k2 = (2*(b - e)**2*(c - f)*np.sqrt(-a*b*e + a*e**2 + b**2*d + b*c**2*e - 2*b*c*e*f - b*d*e + b*e*f**2) + (b**2 - 2*b*e + e**2)*(-a*b + a*e + b*c**2 - 2*b*c*f + b*d + b*f**2 + c**2*e - 2*c*e*f - d*e + e*f**2))/((b - e)**2*(b**2 - 2*b*e + e**2))
    s2 = c + np.sqrt(a + b*(2*(b - e)**2*(c - f)*np.sqrt(-a*b*e + a*e**2 + b**2*d + b*c**2*e - 2*b*c*e*f - b*d*e + b*e*f**2) + (b**2 - 2*b*e + e**2)*(-a*b + a*e + b*c**2 - 2*b*c*f + b*d + b*f**2 + c**2*e - 2*c*e*f - d*e + e*f**2))/((b - e)**2*(b**2 - 2*b*e + e**2)))
    return k1, s1, k2, s2

def get_intersections_ew(STEPS = 5000, zroot1 = Z_ROOT1, zroot2 = Z_ROOT2, zroot3 = Z_ROOT3):
    k = np.linspace(-10, 10, STEPS)
    all_z, stepzise = np.linspace(-zroot3-0.1, 10, STEPS, retstep=True)
    intersections = [(4,0), (0,0)]
    # plt.plot(4,0, 'ro')
    # plt.plot(0,0, 'ro')

    for z in all_z:
        if zroot1-0.01 < abs(z) < zroot1+0.01 or zroot2-0.01 < abs(z) < zroot2+0.01 or zroot3-0.01 < abs(z) < zroot3+0.01 or 1.8-0.035 < abs(z) < 1.8+0.035 or 1.67-0.015 < abs(z) < 1.67+0.015:
            continue
        # plt.plot(k, parabolic_boundary_lines(z, k), color='black')
        # plt.plot(k, -parabolic_boundary_lines(z, k), color='black')
        try:
            a, b, c = parabolic_boundary_lines_coef(z)
            d, e, f = parabolic_boundary_lines_coef(z + stepzise)
            x1, y1, x2, y2 = intersection_parabolas(a, b, c, d, e, f)
            yvalues = parabolic_boundary_lines(z, k)
            # Try 1
            # plt.plot(x1, y1, 'ro')
            # plt.plot(x2, y2, 'bo')
            
            # Try 2
            # plt.plot(x1, abs(y1), 'ro')
                
            # Try 3 and more
            if 0 <= x1 <= 4 and 0 <= abs(y1) < 1:
                # plt.plot(x1, abs(y1), 'ro')
                intersections.append((x1, abs(y1)))
        except Exception as e:
            print('Error at z =', z, e)
            
    return intersections
            
def neg_log_likelihood_ew(params, data):
    mu, variance, s, k = params
    s, k = transform_skew_exkurt_into_positivity_region(s, k, get_intersections_ew())
    likelihoods = edgeworth_expansion(data, mu, variance, s, k)
    return -np.sum(np.log(likelihoods))

def edgeworth_expansion_positivity_constraint(x, mean, variance, skewness, exkurt, fakasawa = False):
    new_skew, new_exkurt = transform_skew_exkurt_into_positivity_region(skewness, exkurt, get_intersections_ew())
    expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(mean, variance, new_skew, new_exkurt), fakasawa)
        
    return expansion

def cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return mean*x + 0.5*variance*x**2 + third_cumulant*x**3/6 + fourth_cumulant*x**4/24

def derivative_cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return mean + variance*x + third_cumulant*x**2/2 + fourth_cumulant*x**3/6

def second_derivative_cgf(x, mean, variance, third_cumulant, fourth_cumulant):
    return variance + third_cumulant*x + fourth_cumulant*x**2/2

def find_saddlepoint(z, mean, variance, third_cumulant, fourth_cumulant):
    """solves K'(t) = z and returns t"""
    if fourth_cumulant == 0 and third_cumulant == 0 and variance == 0:
        return None
    elif fourth_cumulant == 0 and third_cumulant == 0:
        return (z - mean)/variance
    elif fourth_cumulant == 0:
        # can this ever happen?
        sqrt_term = np.sqrt(-2*mean*third_cumulant + 2*third_cumulant*z + variance**2)
        return (sqrt_term - variance)/third_cumulant
        # return (-sqrt_term - variance)/skewness
    else:
        term_with_162 = -162*fourth_cumulant**2*mean + 162*fourth_cumulant**2*z + 162*fourth_cumulant*third_cumulant*variance - 54*third_cumulant**3
        term_with_18 = 18*fourth_cumulant*variance - 9*third_cumulant**2
        term_with_sqrt = np.sqrt(term_with_162**2 + 4*term_with_18**3)
        return 1/(3*np.cbrt(2)*fourth_cumulant) * np.cbrt(term_with_sqrt + term_with_162) - (np.cbrt(2)*term_with_18)/(3*fourth_cumulant*np.cbrt(term_with_sqrt + term_with_162)) - third_cumulant/fourth_cumulant
    
def saddlepoint_approximation(x, mean, variance, third_cumulant, fourth_cumulant):
    # get saddle point for x
    s = find_saddlepoint(x, mean, variance, third_cumulant, fourth_cumulant)
    return 1/np.sqrt(2*np.pi*second_derivative_cgf(s, mean, variance, third_cumulant, fourth_cumulant)) * np.exp(cgf(s, mean, variance, third_cumulant, fourth_cumulant) - s*x)

def two_d_interpolation(x1, y1, x2, y2, d_x, d_y):
    l = (d_x + d_y)/2
    return x1 + (x2 - x1) * l, y1 + (y2 - y1) * l

def find_s_k_from_skewness_exkurt_table(skewness, excess_kurtosis, interpolate = True):
    skew_parameter = None
    exkurt_parameter = None
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
    
    actual_skewness_idx = np.searchsorted(actual_skewness_values, skewness)
    actual_exkurt_idx = np.searchsorted(actual_exkurt_values, excess_kurtosis)
    
    if actual_skewness_idx < len(actual_skewness_values) and actual_exkurt_idx < len(actual_exkurt_values) and actual_skewness_idx < len(s_values[actual_exkurt_idx]):
        if skewness in actual_skewness_values and excess_kurtosis in actual_exkurt_values:
            return s_values[actual_exkurt_idx][actual_skewness_idx], k_values[actual_exkurt_idx][actual_skewness_idx]
        if interpolate:
            s2, k2 = s_values[actual_exkurt_idx][actual_skewness_idx], k_values[actual_exkurt_idx][actual_skewness_idx]
            
            actual_exkurt_idx = actual_exkurt_idx - 1 if actual_exkurt_idx > 0 else 0
            
            # versuche in der Tabelle für Skewness eins nach links zu gehen
            actual_skewness_idx = actual_skewness_idx - 1 if actual_skewness_idx > 0 else 0
            # das klappt aber nicht immer
            if actual_skewness_idx >= len(s_values[actual_exkurt_idx]):
                actual_skewness_idx = len(s_values[actual_exkurt_idx]) - 1
            
            s1, k1 = s_values[actual_exkurt_idx][actual_skewness_idx], k_values[actual_exkurt_idx][actual_skewness_idx]
            
            d_x = skewness - actual_skewness_values[actual_skewness_idx]
            d_y = excess_kurtosis - actual_exkurt_values[actual_exkurt_idx]
            
            return two_d_interpolation(s1, k1, s2, k2, d_x, d_y)
        else:
            return s_values[actual_exkurt_idx][actual_skewness_idx], k_values[actual_exkurt_idx][actual_skewness_idx]
    else:
        return skewness, excess_kurtosis

def numerical_derivative(x, y):
    """
    Approximates the derivative of a function given x and y values.

    Parameters:
    - x: List or numpy array of x values (must be sorted in ascending order).
    - y: List or numpy array of y values (same length as x).

    Returns:
    - deriv: List of derivative values at the midpoints of x.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    deriv = np.zeros(n)

    # Forward difference for the first point
    deriv[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Central difference for interior points
    for i in range(1, n - 1):
        deriv[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    # Backward difference for the last point
    deriv[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return deriv
    
def cornish_fisher_expansion(x, mean, variance, skewness, excess_kurtosis, pdf = True, interpolate = True):
    return_values = []
    s, k = find_s_k_from_skewness_exkurt_table(skewness, excess_kurtosis, interpolate)
    x_values = np.linspace(-10, 10, 1000)
    z_p = norm.cdf(x_values, loc = mean, scale = np.sqrt(variance))
    x_p = z_p + s/6 * (z_p**2 - 1) + k/24 * (z_p**3 - 3*z_p) - s**2/36 * (2*z_p**3 - 5*z_p)
    density = numerical_derivative(x_values, x_p)
    
    x_orig = x # store the original x value and type
    if isinstance(x, int) or isinstance(x, float):
        x = [x]
    
    for single_x in x:
        # find the index i in x_values such that x_values[i] < single_x <= x_values[i+1]
        i = 0
        while x_values[i] < single_x:
            i += 1
        # linear interpolation
        if pdf:
            return_values.append(density[i] + (density[i+1] - density[i])/(x_values[i+1] - x_values[i]) * (single_x - x_values[i]))
        else:
            return_values.append(x_p[i] + (x_p[i+1] - x_p[i])/(x_values[i+1] - x_values[i]) * (single_x - x_values[i]))
    
    if isinstance(x_orig, int) or isinstance(x_orig, float):
        return return_values[0]
    else:
        return return_values