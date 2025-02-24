import sqlite3
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

conn = sqlite3.connect('simulations.db')
c = conn.cursor()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expansion_methods.all_methods import gram_charlier_expansion
from compare_distributions.distances import pdf_to_cdf
from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate

# c.execute('select * from simulations where id = 877')
c.execute('select * from simulations where id = 346638')
simulation = c.fetchone()
cumulants = simulation[16:20]
moments = simulation[20:24]
mu = simulation[1]
kappa = simulation[2]
theta = simulation[3]
sigma = simulation[4]
rho = simulation[5]
v0 = simulation[6]

x = np.linspace(-2, 2, 1000)
expansion = gram_charlier_expansion(x, *cumulants, fakasawa=True)
x_theory, density = compute_density_via_ifft_accurate(mu, kappa, theta, sigma, rho, 1/12)
empirical_cdf = pdf_to_cdf(x, expansion)
theory_cdf = pdf_to_cdf(x_theory, density)
print(theory_cdf)


# Hill Plot
k_min = 10
k_max = 490

x_sorted = sorted(x, reverse=True)
x_theory_sorted = sorted(x_theory, reverse=True)
hill_estimates = []
hill_estimates_theory = []
k_values = range(k_min, k_max+1, 10)

for k in k_values:
    x_tail = x_sorted[:k+1]
    hill_k = (1/k) * sum(np.log(x / x_tail[-1]) for x in x_tail[:-1])
    hill_estimates.append(hill_k)
    
    x_theory_tail = x_theory_sorted[:k+1]
    hill_k = (1/k) * sum(np.log(x_theory / x_theory_tail[-1]) for x_theory in x_theory_tail[:-1])
    hill_estimates_theory.append(hill_k)

plt.plot(k_values, hill_estimates, marker='o', label='Empirical density')
plt.plot(k_values, hill_estimates_theory, marker='x', label='Theoretical densitiy')
plt.xlabel('k (number of extreme values)')
plt.ylabel('Hill estimator (tail index)')
plt.title('Hill Plot')
plt.legend()
plt.show()

# CCDF, empirische Überlebensfunktion + Verlgeich zwischen Paretto und Exponential

# Pareto (heavy-tailed)
alpha = 2.5        # Tail-Index
x_min = 1.0
# Exponential (light-tailed)
lambda_exp = 1.0

x_vals = np.linspace(x_min, 50, 500)

# CCDF der Pareto-Verteilung: P(X>x) = (x_min/x)^alpha, x >= x_min
pareto_ccdf = (x_min / x_vals)**alpha

# CCDF der Exponentialverteilung: P(X>x) = exp(-lambda * x)
exp_ccdf = np.exp(-lambda_exp * x_vals)

plt.figure(figsize=(7,7))
plt.subplot(2,2,1)
plt.plot(x_vals, pareto_ccdf, label=f'Pareto (alpha = {alpha})')
plt.xlabel('x')
plt.ylabel('CCDF = P(X>x)')
plt.xscale('log')
plt.yscale('log')
plt.title('Heavy-tailed distribution (Pareto)')
plt.legend()

plt.subplot(2,2,2)
plt.plot(x_vals, exp_ccdf, color='orange', label=f'Exponential (lambda = {lambda_exp})')
plt.xlabel('x')
plt.ylabel('CCDF = P(X>x)')
plt.xscale('log')
plt.yscale('log')
plt.title('Light-tailed distribution (Exponential)')
plt.legend()

plt.subplot(2,2,3)
plt.plot(x, 1-empirical_cdf, label='empirical CCDF', color = 'red')
plt.xlabel('x')
plt.ylabel('CCDF = P(X>x)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Empirical CCDF')

plt.subplot(2,2,4)
plt.plot(x, 1-theory_cdf, label='theoretical CCDF', color = 'green')
plt.xlabel('x')
plt.ylabel('CCDF = P(X>x)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Theoretical CCDF')

plt.tight_layout()
plt.show()