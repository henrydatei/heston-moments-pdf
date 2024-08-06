import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct
from scipy.optimize import minimize, differential_evolution

STEPS = 1000
Z_START = -10
Z_END = -np.sqrt(3)

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

all_z, stepsize = np.linspace(Z_START, Z_END, STEPS, retstep=True)
intersections = [(4,0), (0,0)]

for z in all_z:
    skew_a1, skew_b1, _, _ = get_positivity_boundary_lines(z)
    skew_a2, skew_b2, _, _ = get_positivity_boundary_lines(z+stepsize)
    inter_x, inter_y = intersection_lines(skew_a1, skew_b1, skew_a2, skew_b2)
    intersections.append((inter_x, inter_y))

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

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)

# Function to generate Gram-Charlier expansion
def gram_charlier_expansion(x, mean, variance, skewness, excess_kurtosis):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(z) * (1 + skewness/6 * hermite_polynomial(3, z) + excess_kurtosis/24 * hermite_polynomial(4, z))

def neg_log_likelihood(params, data):
    mu, variance, s, k = params
    s, k = transform_skew_kurt_into_positivity_region(s, k, intersections)
    likelihoods = gram_charlier_expansion(data, mu, variance, s, k)
    return -np.sum(np.log(likelihoods))

# def log_likelihood_differential_evolution(params, *data):
#     mu, variance, s, k = params
#     s, k = transform_skew_kurt_into_positivity_region(s, k, intersections)
#     data = np.array(data).flatten()
#     likelihoods = gram_charlier_expansion(data, mu, variance, s, k)
#     return -np.sum(np.log(likelihoods))

# Calculate skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_exkurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_exkurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_exkurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_mean, normal_var, normal_skew, normal_exkurt)
print(lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt)
print(t_mean, t_var, t_skew, t_exkurt)
print(nct_mean, nct_var, nct_skew, nct_exkurt)

# plot the positivity boundary
# plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 2, color = 'r')
# plt.plot([lognorm_exkurt], [lognorm_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'b')
# plt.plot([t_exkurt], [t_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'g')
# plt.plot([nct_exkurt], [nct_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'y')
# plt.title('Positivity Boundary of Gram-Charlier Density Function')
# plt.xlabel('Kurtosis')
# plt.ylabel('Skewness')
# plt.legend(['Positivity Boundary', 'Log-Normal', 't', 'NCT'])
# plt.tight_layout()
# plt.show()

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Gram-Charlier expansion with MLE
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

normal_result = minimize(neg_log_likelihood, normal_initial_params, args=(normal_data), method='L-BFGS-B', bounds=normal_bounds)
lognorm_result = minimize(neg_log_likelihood, lognorm_initial_params, args=(lognorm_data), method='L-BFGS-B', bounds=lognorm_bounds)
t_result = minimize(neg_log_likelihood, t_initial_params, args=(t_data), method='L-BFGS-B', bounds=t_bounds)
nct_result = minimize(neg_log_likelihood, nct_initial_params, args=(nct_data), method='L-BFGS-B', bounds=nct_bounds)

# normal_result = differential_evolution(log_likelihood_differential_evolution, normal_bounds, args=(normal_data), strategy='best1bin', maxiter=1000)
if normal_result.success:
    mu, sigma2, skew, exkurt = normal_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu}, sigma^2 = {sigma2}, s = {skew}, k = {exkurt}")
    print(f"Log-likelihood fitted: {neg_log_likelihood([mu, sigma2, skew, exkurt], normal_data)}, Log-likelihood initial: {neg_log_likelihood([normal_mean, normal_var, normal_skew, normal_exkurt], normal_data)}")
    initial_expansion = gram_charlier_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    initial_expansion = [0] * len(x)
    
# lognorm_result = differential_evolution(log_likelihood_differential_evolution, lognorm_bounds, args=(lognorm_data), strategy='best1bin', maxiter=1000)
if lognorm_result.success:
    mu, sigma2, skew, exkurt = lognorm_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu}, sigma^2 = {sigma2}, s = {skew}, k = {exkurt}")
    print(f"Log-likelihood fitted: {neg_log_likelihood([mu, sigma2, skew, exkurt], lognorm_data)}, Log-likelihood initial: {neg_log_likelihood([lognorm_mean, lognorm_var, lognorm_skew, lognorm_exkurt], lognorm_data)}")
    lognorm_expansion = gram_charlier_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    lognorm_expansion = [0] * len(x)

# t_result = differential_evolution(log_likelihood_differential_evolution, t_bounds, args=(t_data), strategy='best1bin', maxiter=1000)
if t_result.success:
    mu, sigma2, skew, exkurt = t_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu}, sigma^2 = {sigma2}, s = {skew}, k = {exkurt}")
    print(f"Log-likelihood fitted: {neg_log_likelihood([mu, sigma2, skew, exkurt], t_data)}, Log-likelihood initial: {neg_log_likelihood([t_mean, t_var, t_skew, t_exkurt], t_data)}")
    t_expansion = gram_charlier_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    t_expansion = [0] * len(x)

# nct_result = differential_evolution(log_likelihood_differential_evolution, nct_bounds, args=(nct_data), strategy='best1bin', maxiter=1000)
if nct_result.success:
    mu, sigma2, skew, exkurt = nct_result.x
    skew, exkurt = transform_skew_kurt_into_positivity_region(skew, exkurt, intersections)
    print(f"Fitted parameters: mu = {mu}, sigma^2 = {sigma2}, s = {skew}, k = {exkurt}")
    print(f"Log-likelihood fitted: {neg_log_likelihood([mu, sigma2, skew, exkurt], nct_data)}, Log-likelihood initial: {neg_log_likelihood([nct_mean, nct_var, nct_skew, nct_exkurt], nct_data)}")
    nct_expansion = gram_charlier_expansion(x, mu, sigma2, skew, exkurt)
else:
    print("Optimization failed.")
    nct_expansion = [0] * len(x)
    
# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, initial_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title('Normal Distribution and Positivity Gram-Charlier Expansion')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title('Log-Normal Distribution and Positivity Gram-Charlier Expansion')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title('t Distribution and Positivity Gram-Charlier Expansion')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title('Non-central t Distribution and Positivity Gram-Charlier Expansion')
plt.legend()

plt.tight_layout()
plt.show()

# Solver Test
initial_params = [1,1,1,1]
print(f'Log-likelihood initial: {-neg_log_likelihood(initial_params, normal_data)}')
plt.plot(x, gram_charlier_expansion(x, *initial_params), 'r--', label='Initial')
plt.plot(x, gram_charlier_expansion(x, normal_mean, normal_var, normal_skew, normal_exkurt), 'g--', label='True')

for method in [
    'Nelder-Mead', 
    'Powell', 
    'CG', 
    'BFGS', 
    # 'Newton-CG', # Jacobian is required for Newton-CG method
    'L-BFGS-B', 
    'TNC', 
    'COBYLA',  
    'SLSQP', 
    'trust-constr', 
    # 'dogleg', # Jacobian is required for Newton-CG method
    # 'trust-ncg', # Jacobian is required for Newton-CG method
    # 'trust-exact', # Jacobian is required for Newton-CG method
    # 'trust-krylov'# # Jacobian is required for Newton-CG method
    ]:
    res = minimize(neg_log_likelihood, initial_params, args=(normal_data), method=method)
    mu, sigma2, skew, exkurt = res.x
    # print(f"Fitted parameters: mu = {mu}, sigma^2 = {sigma2}, s = {skew}, k = {exkurt}")
    print(f"{method}: Log-likelihood fitted: {-neg_log_likelihood([mu, sigma2, skew, exkurt], normal_data)}")
    plt.plot(x, gram_charlier_expansion(x, mu, sigma2, skew, exkurt), label=method)

plt.yscale('log')    
plt.legend()
plt.tight_layout()
plt.show()