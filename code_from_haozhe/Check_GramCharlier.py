import numpy as np
from GramCharlier_expansion import Expansion_GramCharlier
import matplotlib.pyplot as plt

# plot the approximate density function

x = np.linspace(-2.0, 2.0, 1000)
cumulant = np.array([-0.01401158, 0.01647651, -0.0006112, 0.00093973])

f_x = Expansion_GramCharlier(cumulant)
plt.figure(figsize=(12, 8))
plt.grid()
plt.xlabel("x")
plt.ylabel("$f(x)$")
plt.plot(x, f_x, '--b')
plt.savefig('ApproxiDensityFunction.pdf') 
plt.show()