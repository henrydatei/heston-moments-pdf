# Moments, Central Moments and Cumulants

The $r$-th moment $\mu_r'$ of a random variable $X$ is defined as $\mathbb{E}(X^r)$.

The $r$-th central moment $\mu_r$ of a random variable $X$ is defined as $\mathbb{E}((X-\mu)^r)$, where $\mu$ is the mean of $X$.

The $r$-th cumulant $\kappa_r$ of a random variable $X$ is defined as the coefficient of $t^r$ in the logarithm of the moment generating function of $X$. The moment generating function of $X$ is defined as $M_X(t) = \mathbb{E}(\exp(tX))$. The cumulant generating function of $X$ is defined as $K_X(t) = \log(M_X(t))$.

Mean $\mu$ is the first moment. $$\mu = \mathbb{E}(X) = \kappa_1$$

Variance $\sigma^2$ is the second central moment. $$\sigma^2 = \mathbb{E}[(X-\mu)^2] = \kappa_2$$

Skewness $\gamma_1$ is the standardized third central moment. $$\gamma_1 = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3} = \frac{\kappa_3}{\kappa_2^{3/2}}$$

Kurtosis $\gamma_2$ is the standardized fourth central moment. $$\gamma_2 = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4} = \frac{\kappa_4}{\kappa_2^2} - 3$$

Excess kurtosis $\gamma_2^*$ is the kurtosis minus 3. $$\gamma_2^* = \gamma_2 - 3 = \frac{\kappa_4}{\kappa_2^2}$$

From $\mu$, $\sigma^2$, $\gamma_1$ and $\gamma_2^*$, we can calculate $\kappa_1$, $\kappa_2$, $\kappa_3$ and $\kappa_4$.
$$\begin{align}
\kappa_1 &= \mu \\
\kappa_2 &= \sigma^2 \\
\kappa_3 &= \gamma_1\cdot\left(\sigma^2\right)^{3/2} \\
\kappa_4 &= \gamma_2^*\cdot\left(\sigma^2\right)^2
\end{align}$$