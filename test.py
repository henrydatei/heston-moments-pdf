import numpy as np

from expansion_methods.gc_expansion import gram_charlier_expansion

x = np.linspace(-3, 3, 100)
y = gram_charlier_expansion(x, 0, 1, 0, 0)

print(y)