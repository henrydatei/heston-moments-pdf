import numpy as np
import matplotlib.pyplot as plt

# Heston model parameters
S0 = 100.0  # Initial stock price
v0 = 0.04   # Initial variance (volatility squared)
mu = 0.05   # Drift (expected return)
kappa = 2.0 # Rate of mean reversion
theta = 0.04 # Long-term variance
sigma = 0.5  # Volatility of volatility
rho = -0.7  # Correlation between two Brownian motions
T = 1.0     # Time horizon
N = 1000    # Number of time steps
n_paths = 10  # Number of simulated paths

# Seed for reproducibility
np.random.seed(42)

def heston_euler(S0, v0, mu, kappa, theta, sigma, rho, T, N, n_paths):
    dt = T / N # Time step size
    
    # Pre-allocate arrays for asset prices and variances
    S_paths = np.zeros((n_paths, N + 1))
    v_paths = np.zeros((n_paths, N + 1))

    # Initial conditions
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    # Simulate correlated Brownian motions
    dW1 = np.random.normal(0, np.sqrt(dt), (n_paths, N))
    dW2 = np.random.normal(0, np.sqrt(dt), (n_paths, N))

    # Correlate the Brownian motions using Cholesky decomposition
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2

    # Euler-Maruyama discretization
    for i in range(N):
        # Ensure non-negative variance (CIR process constraint)
        v_paths[:, i + 1] = np.maximum(v_paths[:, i] + kappa * (theta - v_paths[:, i]) * dt + sigma * np.sqrt(v_paths[:, i]) * dW2[:, i], 0)
        S_paths[:, i + 1] = S_paths[:, i] * np.exp((mu - 0.5 * v_paths[:, i]) * dt + np.sqrt(v_paths[:, i]) * dW1[:, i])
    
    return S_paths

S_paths = heston_euler(S0, v0, mu, kappa, theta, sigma, rho, T, N, n_paths)

# Plot some sample paths for stock prices
plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(np.linspace(0, T, N + 1), S_paths[i, :], lw=1.5)
plt.title("Simulated Asset Price Paths under the Heston Model")
plt.xlabel("Time")
plt.ylabel("Asset Price")
plt.grid(True)
plt.show()
