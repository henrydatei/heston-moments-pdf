import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct
from scipy.optimize import minimize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from all_methods import neg_log_likelihood_gc, get_intersections_gc, transform_skew_exkurt_into_positivity_region, gram_charlier_expansion_positivity_constraint, gram_charlier_expansion, scipy_mvsek_to_cumulants

# plot the positivity boundary
intersections = get_intersections_gc()
plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 2, color = 'r')
plt.title('Positivity Boundary of Gram-Charlier Density Function')
plt.xlabel('Excess Kurtosis')
plt.ylabel('Skewness')
plt.tight_layout()
plt.show()

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# scipy set seed
np.random.seed(0)

# Apply Gram-Charlier expansion with MLE
N_SAMPLES = 1000
normal_data = norm.rvs(size=N_SAMPLES)
lognorm_data = lognorm.rvs(0.5, size=N_SAMPLES)
t_data = t.rvs(5, size=N_SAMPLES)
nct_data = nct.rvs(5, 0.5, size=N_SAMPLES)

normal_expansion = gram_charlier_expansion_positivity_constraint(x, *norm.stats(moments='mvsk'))
lognorm_expansion = gram_charlier_expansion_positivity_constraint(x, *lognorm.stats(0.5, moments = 'mvsk'))
t_expansion = gram_charlier_expansion_positivity_constraint(x, *t.stats(5, moments = 'mvsk'))
nct_expansion = gram_charlier_expansion_positivity_constraint(x, *nct.stats(5, 0.5, moments = 'mvsk'))
    
# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title(f'Normal Distribution and Positivity Gram-Charlier Expansion (n = {N_SAMPLES})')
plt.legend()

# Plot Skewed distribution and its expansion
plt.subplot(4, 1, 2)
plt.plot(x, lognorm.pdf(x, 0.5), 'r--', label='Log-Normal PDF')
plt.plot(x, lognorm_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title(f'Log-Normal Distribution and Positivity Gram-Charlier Expansion (n = {N_SAMPLES})')
plt.legend()

# Plot Heavy-tailed distribution and its expansion
plt.subplot(4, 1, 3)
plt.plot(x, t.pdf(x, 5), 'r--', label='t PDF')
plt.plot(x, t_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title(f't Distribution and Positivity Gram-Charlier Expansion (n = {N_SAMPLES})')
plt.legend()

# Plot Non-central t distribution and its expansion
plt.subplot(4, 1, 4)
plt.plot(x, nct.pdf(x, 5, 0.5), 'r--', label='NCT PDF')
plt.plot(x, nct_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
plt.title(f'Non-central t Distribution and Positivity Gram-Charlier Expansion (n = {N_SAMPLES})')
plt.legend()

plt.tight_layout()
plt.show()

# Solver Test
initial_params = [1,1,1,1] # mu, sigma2, skew, exkurt
print(f'Log-likelihood initial: {-neg_log_likelihood_gc(initial_params, normal_data)}')
plt.plot(x, gram_charlier_expansion(x, *scipy_mvsek_to_cumulants(*initial_params)), 'r--', label='Initial')
plt.plot(x, gram_charlier_expansion(x, *scipy_mvsek_to_cumulants(*norm.stats(moments = 'mvsk'))), 'g--', label='True')

for method in [
    'Nelder-Mead', 
    'Powell', 
    'CG', 
    'BFGS', 
    # 'Newton-CG', # Jacobian is required
    'L-BFGS-B', 
    'TNC', 
    'COBYLA',  
    'SLSQP', 
    'trust-constr', 
    # 'dogleg', # Jacobian is required
    # 'trust-ncg', # Jacobian is required
    # 'trust-exact', # Jacobian is required
    # 'trust-krylov'# # Jacobian is required
    ]:
    res = minimize(neg_log_likelihood_gc, initial_params, args=(normal_data), method=method)
    mu, sigma2, skew, exkurt = res.x
    print(f"{method}: Log-likelihood fitted: {-neg_log_likelihood_gc([mu, sigma2, skew, exkurt], normal_data):.4f}")
    skew, exkurt = transform_skew_exkurt_into_positivity_region(skew, exkurt, get_intersections_gc())
    print(f"Fitted parameters: {mu:.4f}, {sigma2:.4f}, {skew:.4f}, {exkurt:.4f}")
    plt.plot(x, gram_charlier_expansion(x, *scipy_mvsek_to_cumulants(mu, sigma2, skew, exkurt)), label=method)

plt.title('Optimization Methods Comparison')   
plt.legend()
plt.tight_layout()
plt.show()