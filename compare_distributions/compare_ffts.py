import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heston_model_properties.theoretical_density import compute_density_via_ifft_accurate
from code_from_haozhe.HestonDensity_FFT import HestonChfDensity_FFT_Gatheral

c = sqlite3.connect('simulations.db')
cursor = c.cursor()
random_simulation = cursor.execute('SELECT * FROM simulations WHERE id = 57940').fetchone()
c.close()

mu = random_simulation[1]
kappa = random_simulation[2]
theta = random_simulation[3]
sigma = random_simulation[4]
rho = random_simulation[5]
v0 = random_simulation[6]

try:
    x_density, density = compute_density_via_ifft_accurate(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, t=1/12)
except Exception as e:
    print(f'Henry Error: {e}')
    x_density, density = None, None

try:
    f_x = HestonChfDensity_FFT_Gatheral(mu = mu, kappa = kappa, theta = theta, sigma = sigma, rho=rho, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = v0, conditional=False)
except Exception as e:
    print(f'Haozhe Error: {e}')
    f_x = None

x = np.linspace(-2, 2, 1000)

if (x_density is not None and density is not None) or (f_x is not None):
    if x_density is not None and density is not None:
        plt.plot(x_density, density, 'y-', label='Theoretical Density Henry')
    if f_x is not None:
        plt.plot(x, f_x, 'b-', label='Theoretical Density Haozhe')

    plt.legend()

    plt.tight_layout()
    plt.xlim(-2, 2)
    plt.ylim(0, 5)
    plt.show()