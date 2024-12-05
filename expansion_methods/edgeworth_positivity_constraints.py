import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from all_methods import edgeworth_expansion_positivity_constraint, get_intersections_ew, parabolic_boundary_lines

# intersections = sorted(intersections, key=lambda x: x[0])
# print(intersections[0], intersections[1], intersections[-2], intersections[-1])

# Plot lines
for z in np.linspace(-10, 0, 1000):
    plt.plot(np.linspace(-10, 10, 1000), parabolic_boundary_lines(z, np.linspace(-10, 10, 1000)), color='black')
intersections = get_intersections_ew()
plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 1, color = 'r')
plt.xlim(-1, 5)
plt.ylim(0, 2)
plt.xlabel('Excess Kurtosis')
plt.ylabel('Skewness')
plt.title('Boundary Lines of Positivity Region')
plt.tight_layout()
plt.show()

# plot the positivity boundary
intersections = get_intersections_ew()
plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 2, color = 'r')
plt.title('Positivity Boundary of Edgeworth Density Function')
plt.xlabel('Excess Kurtosis')
plt.ylabel('Skewness')
plt.tight_layout()
plt.show()

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

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