import numpy as np

def cornish_fisher_expansion(x, skew, kurt):
    """Returns the Cornish-Fisher expansion of a random variable x with skewness skew and kurtosis kurt.
    Source: "Option Pricing Under Skewness and Kurtosis Using a Cornish–Fisher Expansion"

    Args:
        x (float): The value of the random variable.
        skew (float): The skewness of the random variable.
        kurt (float): The kurtosis of the random variable.
    """
    a = (-skew)/(kurt/8 - (skew**2)/3)
    b = kurt/24 - (skew**2)/18
    p = (1 - kurt/8 + 5*skew**2/36)/(kurt/24 - (skew**2)/18) - 1/3 * ((skew**2)/36)/((kurt/24 - (skew**2)/18)**2)
    q = (-skew)/(kurt/8 - (skew**2)/3) - 1/18 * (skew * (1 - kurt/8 + 5*skew**2/36))/((kurt/24 - skew**2/18)**2) - 2/27 * (skew**3/216)/((kurt/24 - skew**2/18)**3)
    z = a/3 + np.cbrt((-q + x/b + np.sqrt((q-x/b)**2 + 4/27*p**3))/2) + np.cbrt((-q + x/b - np.sqrt((q-x/b)**2 + 4/27*p**3))/2)
    return 1/np.sqrt(2*np.pi) * np.exp(-z**2/2)/(z**2 * (kurt/8 - skew**2/6) + z*skew/3 + 1 - kurt/8 + 5*skew**2/36)

print(cornish_fisher_expansion(0, 0, 0.1))