import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def scipy_moments_to_cumulants(mean, variance, skewness, excess_kurtosis):
    return mean, variance, skewness*variance**1.5, excess_kurtosis*variance**2

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

def edgeworth_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(z) * (1 + third_cumulant/6 * hermite_polynomial(3, z) + fourth_cumulant/24 * hermite_polynomial(4, z) + third_cumulant**2/72 * hermite_polynomial(6, z))

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

def get_intersections(STEPS = 1000, Z_START = -10, Z_END = -np.sqrt(3)):
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

def neg_log_likelihood(params, data):
    mu, variance, s, k = params
    s, k = transform_skew_kurt_into_positivity_region(s, k, get_intersections())
    likelihoods = gram_charlier_expansion(data, mu, variance, s, k)
    return -np.sum(np.log(likelihoods))

def transform_skew_kurt_into_positivity_region(skew, kurt, intersections):
    skew_sign = np.sign(skew)
    skew = abs(skew)
    new_kurt = logistic_map(kurt, 0, 4)

    if new_kurt == 4:
        return 0, 4
    
    # find i such that intersections[i][0] < new_kurt <= intersections[i+1][0]
    for i in range(len(intersections)-1):
        if intersections[i][0] < new_kurt <= intersections[i+1][0]:
            break

    k_i, s_i = intersections[i]
    k_i2, s_i2 = intersections[i+1]
    a_i = (s_i * k_i2 - k_i * s_i2)/(k_i2 - k_i)
    b_i = (s_i2 - s_i)/(k_i2 - k_i)
    s_u = a_i + b_i * new_kurt
    s_l = -s_u
    # print(i, k_i, s_i, k_i2, s_i2, a_i, b_i, s_u, s_l)

    new_skew = logistic_map(skew, s_l, s_u)
    new_skew = skew_sign * new_skew

    return new_skew, new_kurt

def gram_charlier_expansion_positivity_constraint(x, mean, variance, skewness, exkurt):
    initial_params = [mean, variance, skewness, exkurt]
    bounds = [(min(x)-1, max(x)+1), (0.1, 10), (-10, 10), (-10, 10)]
    result = minimize(neg_log_likelihood, initial_params, args=(x), method='Powell', bounds=bounds)
    
    if result.success:
        mu, sigma2, skew, exkurt = result.x
        skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, get_intersections())
        # print(f"Fitted parameters: mu = {mu:.4f}, sigma^2 = {sigma2:.4f}, skew = {skew:.4f}, exkurt = {exkurt:.4f}")
        # print(f"Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], x):.4f}, Log-likelihood initial: {-neg_log_likelihood([mean, variance, skewness, exkurt], x):.4f}")
        expansion = gram_charlier_expansion(x, mu, sigma2, skew, exkurt)
    else:
        print("Optimization failed.")
        expansion = [0] * len(x)
        
    return expansion