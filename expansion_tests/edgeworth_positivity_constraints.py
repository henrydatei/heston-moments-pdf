import numpy as np
import cmath
import matplotlib.pyplot as plt
import random
from scipy.stats import norm, lognorm, t, nct
from scipy.optimize import minimize

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)

i = 1j

# These are the roots of He_6(z), each root is symmetric about the real axis, np.real() is nessary to remove the imaginary part which comes from numerical error
Z_ROOT1 = np.real(cmath.sqrt(5 - (5**(2/3) * (1 + i * np.sqrt(3)) / (2 * (2 + i * np.sqrt(6)))**(1/3)) - ((1 - i * np.sqrt(3)) * (5 * (2 + i * np.sqrt(6)))**(1/3) / 2**(2/3))))

Z_ROOT2 = np.real(cmath.sqrt(5 - (5**(2/3) * (1 - i * np.sqrt(3)) / (2 * (2 + i * np.sqrt(6)))**(1/3)) - ((1 + i * np.sqrt(3)) * (5 * (2 + i * np.sqrt(6)))**(1/3) / 2**(2/3))))

Z_ROOT3 = np.real(cmath.sqrt(5 + (10**(2/3) / (2 + i * np.sqrt(6))**(1/3)) + ((10 * (2 + i * np.sqrt(6)))**(1/3))))

print(Z_ROOT1, Z_ROOT2, Z_ROOT3)

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

STEPS = 5000
k = np.linspace(-10, 10, STEPS)

# plt.figure(figsize=(8, 7))

# plt.subplot(4, 2, 1)
# all_z = np.linspace(-10, -Z_ROOT3, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (-infinity, -3.32...)')

# plt.subplot(4, 2, 2)
# all_z = np.linspace(Z_ROOT3, 10, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (3.32..., infinity)')

# plt.subplot(4, 2, 3)
# all_z = np.linspace(-Z_ROOT3, -Z_ROOT2, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (-3.32..., -1.89...)')

# plt.subplot(4, 2, 4)
# all_z = np.linspace(Z_ROOT2, Z_ROOT3, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (1.89..., 3.32...)')

# plt.subplot(4, 2, 5)
# all_z = np.linspace(-Z_ROOT2, -Z_ROOT1, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (-1.89..., -0.62...)')

# plt.subplot(4, 2, 6)
# all_z = np.linspace(Z_ROOT1, Z_ROOT2, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (0.62..., 1.89...)')

# plt.subplot(4, 2, 7)
# all_z = np.linspace(-Z_ROOT1, Z_ROOT1, STEPS)
# for z in all_z:
#     if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
#         continue
#     plt.plot(k, parabolic_boundary_lines(z, k), color='black')
#     plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
# plt.xlabel('excess kurtosis')
# plt.ylabel('skewness')
# plt.xlim(-10, 10)
# plt.ylim(-5, 5)
# plt.title('Boundary lines, x in (-0.62..., 0.62...)')

# plt.tight_layout()
# plt.show()


all_z, stepzise = np.linspace(-Z_ROOT3-0.1, 10, STEPS, retstep=True)
intersections = [(4,0), (0,0)]
plt.plot(4,0, 'ro')
plt.plot(0,0, 'ro')

for z in all_z:
    if Z_ROOT1-0.01 < abs(z) < Z_ROOT1+0.01 or Z_ROOT2-0.01 < abs(z) < Z_ROOT2+0.01 or Z_ROOT3-0.01 < abs(z) < Z_ROOT3+0.01 or 1.8-0.035 < abs(z) < 1.8+0.035 or 1.67-0.015 < abs(z) < 1.67+0.015:
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
            plt.plot(x1, abs(y1), 'ro')
            intersections.append((x1, abs(y1)))
    except Exception as e:
        print('Error at z =', z, e)
plt.xlabel('excess kurtosis')
plt.ylabel('skewness')
# plt.xlim(-1, 5)
# plt.ylim(0, 0.8)
plt.title('Boundary lines of the Edgeworth expansion')

plt.tight_layout()
plt.show()

# intersections = sorted(intersections, key=lambda x: x[0])
# print(intersections[0], intersections[1], intersections[-2], intersections[-1])

def logistic_map(x, a,b):
    return a + (b-a)/(1+np.exp(-x))

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

def edgeworth_expansion(x, mean, variance, skewness, kurtosis):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(x, loc = mean, scale = np.sqrt(variance)) * (1 + skewness/6 * hermite_polynomial(3, z) + kurtosis/24 * hermite_polynomial(4, z) + skewness**2/72 * hermite_polynomial(6, z))

def neg_log_likelihood(params, data):
    mu, variance, s, k = params
    s, k = transform_skew_kurt_into_positivity_region(s, k, intersections)
    likelihoods = edgeworth_expansion(data, mu, variance, s, k)
    return -np.sum(np.log(likelihoods))

# Calculate theoretical skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_exkurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_exkurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_exkurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_mean, normal_var, normal_skew, normal_exkurt)
print(lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt)
print(t_mean, t_var, t_skew, t_exkurt)
print(nct_mean, nct_var, nct_skew, nct_exkurt)

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Edgeworth expansion with MLE
normal_data = norm.rvs(size=1000)
lognorm_data = lognorm.rvs(0.5, size=1000)
t_data = t.rvs(5, size=1000)
nct_data = nct.rvs(5, 0.5, size=1000)

normal_initial_params = [normal_mean, normal_var, normal_skew, normal_exkurt]
lognorm_initial_params = [lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt]
t_initial_params = [t_mean, t_var, t_skew, t_exkurt]
nct_initial_params = [nct_mean, nct_var, nct_skew, nct_exkurt]

normal_bounds = [(min(normal_data)-1, max(normal_data)+1), (0.1, 10), (-10, 10), (-10, 10)]
lognorm_bounds = [(min(lognorm_data)-1, max(lognorm_data)+1), (0.1, 10), (-10, 10), (-10, 10)]
t_bounds = [(min(t_data)-1, max(t_data)+1), (0.1, 10), (-10, 10), (-10, 10)]
nct_bounds = [(min(nct_data)-1, max(nct_data)+1), (0.1, 10), (-10, 10), (-10, 10)]

normal_result = minimize(neg_log_likelihood, normal_initial_params, args=(normal_data), method='Powell', bounds=normal_bounds)
lognorm_result = minimize(neg_log_likelihood, lognorm_initial_params, args=(lognorm_data), method='Powell', bounds=lognorm_bounds)
t_result = minimize(neg_log_likelihood, t_initial_params, args=(t_data), method='Powell', bounds=t_bounds)
nct_result = minimize(neg_log_likelihood, nct_initial_params, args=(nct_data), method='Powell', bounds=nct_bounds)

if normal_result.success:
    mu, sigma2, skew, exkurt = normal_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu:.4f}, sigma^2 = {sigma2:.4f}, s = {skew:.4f}, k = {exkurt:.4f}")
    print(f"Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], normal_data):.4f}, Log-likelihood initial: {-neg_log_likelihood([normal_mean, normal_var, normal_skew, normal_exkurt], normal_data):.4f}")
    normal_expansion = edgeworth_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    normal_expansion = [0] * len(x)
    
if lognorm_result.success:
    mu, sigma2, skew, exkurt = lognorm_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu:.4f}, sigma^2 = {sigma2:.4f}, s = {skew:.4f}, k = {exkurt:.4f}")
    print(f"Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], lognorm_data):.4f}, Log-likelihood initial: {-neg_log_likelihood([lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt], lognorm_data):.4f}")
    lognorm_expansion = edgeworth_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    lognorm_expansion = [0] * len(x)

if t_result.success:
    mu, sigma2, skew, exkurt = t_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu:.4f}, sigma^2 = {sigma2:.4f}, s = {skew:.4f}, k = {exkurt:.4f}")
    print(f"Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], t_data):.4f}, Log-likelihood initial: {-neg_log_likelihood([t_mean, t_var, t_skew, t_exkurt], t_data):.4f}")
    t_expansion = edgeworth_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    t_expansion = [0] * len(x)
    
if nct_result.success:
    mu, sigma2, skew, exkurt = nct_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu:.4f}, sigma^2 = {sigma2:.4f}, s = {skew:.4f}, k = {exkurt:.4f}")
    print(f"Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], nct_data):.4f}, Log-likelihood initial: {-neg_log_likelihood([nct_mean, nct_var, nct_skew, nct_exkurt], nct_data):.4f}")
    nct_expansion = edgeworth_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    nct_expansion = [0] * len(x)
    
# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Positivity Edgeworth Expansion')
plt.title('Normal Distribution and Positivity Edgeworth Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Positivity Edgeworth Expansion')
plt.title('Log-Normal Distribution and Positivity Edgeworth Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Positivity Edgeworth Expansion')
plt.title('t Distribution and Positivity Edgeworth Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Positivity Edgeworth Expansion')
plt.title('Non-central t Distribution and Positivity Edgeworth Expansion')
plt.legend()

plt.tight_layout()
plt.show()