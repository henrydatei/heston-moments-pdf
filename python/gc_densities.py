import numpy as np
import matplotlib.pyplot as plt

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

s1, k1 = transform_skew_kurt_into_positivity_region(1.7501896550697178, 5.898445673784778, intersections)
s2, k2 = transform_skew_kurt_into_positivity_region(0, 6, intersections)
print(s1, k1)
print(s2, k2)

# plot the positivity boundary
plt.plot([x[0] for x in intersections], [x[1] for x in intersections], linestyle = 'None', marker = 'o', markersize = 2, color = 'r')
plt.plot([k1], [s1], linestyle = 'None', marker = 'o', markersize = 5, color = 'b')
plt.plot([k2], [s2], linestyle = 'None', marker = 'o', markersize = 5, color = 'g')
plt.title('Positivity Boundary of Gram-Charlier Density Function')
plt.xlabel('Kurtosis')
plt.ylabel('Skewness')
plt.tight_layout()
plt.show()