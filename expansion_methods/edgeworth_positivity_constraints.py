import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from all_methods import edgeworth_expansion_positivity_constraint

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

# intersections = sorted(intersections, key=lambda x: x[0])
# print(intersections[0], intersections[1], intersections[-2], intersections[-1])

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

normal_expansion = edgeworth_expansion_positivity_constraint(x, *norm.stats(moments='mvsk'))
lognorm_expansion = edgeworth_expansion_positivity_constraint(x, *lognorm.stats(0.5, moments = 'mvsk'))
t_expansion = edgeworth_expansion_positivity_constraint(x, *t.stats(5, moments = 'mvsk'))
nct_expansion = edgeworth_expansion_positivity_constraint(x, *nct.stats(5, 0.5, moments = 'mvsk'))
    
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