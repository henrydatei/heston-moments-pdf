import numpy as np
import cmath
import matplotlib.pyplot as plt
import random

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

intersections = sorted(intersections, key=lambda x: x[0])
print(intersections[0], intersections[1], intersections[-2], intersections[-1])