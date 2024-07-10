# heston-moments-pdf
Diploma thesis about the Heston SV Model, theoretical and realized moments &amp; pdf expansion methods

Heston Process ($X_t$ process from "Simulating the Cox–Ingersoll–Ross and Heston processes: matching the first four moments"):
$$\begin{align}
\mathrm{d}S_t &= \mu S_t\mathrm{d}t + \sqrt{v_t}S_t\mathrm{d}W_t^S \\
\mathrm{d}X_t = \mathrm{d}\log(S_t) &= \left(\mu-\frac{1}{2}v_t\right)\mathrm{d}t + \sqrt{v_t}\mathrm{d}W_t^S \\
\mathrm{d}v_t &= \kappa(\theta-v_t)\mathrm{d}t + \sigma\sqrt{v_t}\mathrm{d}W_t^v \\
\mathbb{E}(\mathrm{d}W_t^S\mathrm{d}W_t^v) &= \rho\mathrm{d}t
\end{align}$$
with
- $S_t$ price
- $\mu$ drift
- $v_t$ variance
- $\kappa$ speed of mean reversion
- $\theta$ long-term-variance
- $\sigma$ volatility of variance
- $W_t^S$, $W_t^v$ Wiener Processes

From Heston Process are already known (add citations):
- theoretical moments
- theoretical distribution
- theoretical quantiles

For log-returns $r_t$ in "Distributional properties of continuous time processes: from CIR to bates"

Cumulants of $R_t$, continousely compounded return, in "The Skewness Implied in the Heston Model and Its Application"

Moments of $\frac{S_{t+1}}{S_t}$ in "Estimating Option Prices with Heston’s Stochastic Volatility Model"

both papers use Mathematica to get exact formulas, and suggest using characteristic function
- Wikipedia: If a random variable $X$ has moments up to $k$-th order, then the characteristic function $\phi_X$ is $k$ times continuously differentiable on the entire real line. In this case $\mathbb{E}(X^k)=i^{-k}\phi_X^{(k)}(0)$

Simulation of the process with discretisation gives realised moments (Haozhen works on that) and realised quantiles

Realised Moments + Expansion Method $\to$ pdf/cdf $\to$ compare to theoretical distribution
- which expansion method works best?

Comparing realised moments and theoretical moments was done by Neuberger & Payne (2021)
- with short term skew + kurtosis, long term skew and kurtosis are precisely estimatable