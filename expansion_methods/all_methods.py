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

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)
    
def gram_charlier_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + third_cumulant/(6 * variance**1.5) * hermite_polynomial(3, z) + fourth_cumulant/(24 * variance**2) * hermite_polynomial(4, z))

def edgeworth_expansion(x, mean, variance, third_cumulant, fourth_cumulant):
    z = (x - mean) / np.sqrt(variance)
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

def gram_charlier_expansion_positivity_constraint(x, mean, variance, skewness, exkurt):
    new_skew, new_exkurt = transform_skew_exkurt_into_positivity_region(skewness, exkurt, get_intersections_gc())
    expansion = gram_charlier_expansion(x, *scipy_mvsek_to_cumulants(mean, variance, new_skew, new_exkurt))
        
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

def edgeworth_expansion_positivity_constraint(x, mean, variance, skewness, exkurt):
    new_skew, new_exkurt = transform_skew_exkurt_into_positivity_region(skewness, exkurt, get_intersections_ew())
    expansion = edgeworth_expansion(x, *scipy_mvsek_to_cumulants(mean, variance, new_skew, new_exkurt))
        
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