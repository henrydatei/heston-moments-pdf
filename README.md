# heston-moments-pdf
Diploma thesis about the Heston SV Model, theoretical and realized moments &amp; pdf expansion methods

Heston Process:
$$\begin{align}
\mathrm{d}S_t &= \mu S_t\mathrm{d}t + \sqrt{v_t}S_t\mathrm{d}W_t^S \\
\mathrm{d}v_t &= \kappa(\theta-v_t)\mathrm{d}t + \sigma\sqrt{v_t}\mathrm{d}W_t^v \\
\mathbb{E}(W_t^SW_t^v) &= \rho
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

Simulation of the process with discretisation gives realised moments (Haozhen works on that) and realised quantiles

Realised Moments + Expansion Method $\to$ pdf/cdf $\to$ compare to theoretical distribution
- which expansion method works best?

Comparing realised quantiles and theoretical quantiles was done by Neuberger & Payne