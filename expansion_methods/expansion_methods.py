import numpy as np
from scipy.stats import norm

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)
    
def gram_charlier_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/6 * hermite_polynomial(3, z) + fourth_cumulant/24 * hermite_polynomial(4, z))

def scipy_moments_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

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

def edgeworth_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(z) * (1 + third_cumulant/6 * hermite_polynomial(3, z) + fourth_cumulant/24 * hermite_polynomial(4, z) + third_cumulant**2/72 * hermite_polynomial(6, z))