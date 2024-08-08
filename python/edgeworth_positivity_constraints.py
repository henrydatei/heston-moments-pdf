import numpy as np
import cmath
import matplotlib.pyplot as plt

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

STEPS = 1000
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


all_z = np.linspace(-10, 10, STEPS)
for z in all_z:
    if abs(z) in [Z_ROOT1, Z_ROOT2, Z_ROOT3]:
        continue
    plt.plot(k, parabolic_boundary_lines(z, k), color='black')
    plt.plot(k, -parabolic_boundary_lines(z, k), color='black')    
plt.xlabel('excess kurtosis')
plt.ylabel('skewness')
plt.xlim(-1, 5)
plt.ylim(-1, 1)
plt.title('Boundary lines of the Edgeworth expansion')

plt.tight_layout()
plt.show()