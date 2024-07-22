import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t, nct

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

    return new_skew, new_kurt

def hermite_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return x * hermite_polynomial(n-1, x) - (n-1) * hermite_polynomial(n-2, x)

# Function to generate Gram-Charlier expansion
def gram_charlier_expansion(x, mean, variance, skewness, kurtosis):
    z = (x - mean) / np.sqrt(variance)
    return norm.pdf(z) * (1 + skewness/6 * hermite_polynomial(3, z) + kurtosis/24 * hermite_polynomial(4, z))

# Calculate skewness and kurtosis
normal_mean, normal_var, normal_skew, normal_kurt = norm.stats(moments='mvsk')
lognorm_mean, lognorm_var, lognorm_skew, lognorm_kurt = lognorm.stats(0.5, moments = 'mvsk')
t_mean, t_var, t_skew, t_kurt = t.stats(5, moments = 'mvsk')
nct_mean, nct_var, nct_skew, nct_kurt = nct.stats(5, 0.5, moments = 'mvsk')

print(normal_skew, normal_kurt)
print(lognorm_skew, lognorm_kurt)
print(t_skew, t_kurt)
print(nct_skew, nct_kurt)

lognorm_skew, lognorm_kurt = transform_skew_kurt_into_positivity_region(lognorm_skew, lognorm_kurt, intersections)
t_skew, t_kurt = transform_skew_kurt_into_positivity_region(t_skew, t_kurt, intersections)
nct_skew, nct_kurt = transform_skew_kurt_into_positivity_region(nct_skew, nct_kurt, intersections)
print(lognorm_skew, lognorm_kurt)
print(t_skew, t_kurt)
print(nct_skew, nct_kurt)

# plot the positivity boundary
plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 2, color = 'r')
plt.plot([lognorm_kurt], [lognorm_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'b')
plt.plot([t_kurt], [t_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'g')
plt.plot([nct_kurt], [nct_skew], linestyle = 'None', marker = 'o', markersize = 5, color = 'y')
plt.title('Positivity Boundary of Gram-Charlier Density Function')
plt.xlabel('Kurtosis')
plt.ylabel('Skewness')
plt.legend(['Positivity Boundary', 'Log-Normal', 't', 'NCT'])
plt.tight_layout()
plt.show()

# Define x range for plotting
x = np.linspace(-5, 5, 1000)

# Apply Gram-Charlier expansion
normal_expansion = gram_charlier_expansion(x, normal_mean, normal_var, normal_skew, normal_kurt)
lognorm_expansion = gram_charlier_expansion(x, lognorm_mean, lognorm_var, lognorm_skew, lognorm_kurt)
t_expansion = gram_charlier_expansion(x, t_mean, t_var, t_skew, t_kurt)
nct_expansion = gram_charlier_expansion(x, nct_mean, nct_var, nct_skew, nct_kurt)

# Plotting
plt.figure(figsize=(8, 7))

# Plot Normal distribution and its expansion
plt.subplot(4, 1, 1)
plt.plot(x, norm.pdf(x), 'r--', label='Normal PDF')
plt.plot(x, normal_expansion, 'b-', label='Positivity Gram-Charlier Expansion')
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
