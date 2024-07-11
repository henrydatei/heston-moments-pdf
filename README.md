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

---

From Heston Process are already known (add citations):
- theoretical moments
- theoretical distribution
- theoretical quantiles

For log-returns $r_t=X_t-X_0$ in "Distributional properties of continuous time processes: from CIR to bates" $\to$ only for jump diffusion process!

Cumulants of $R_t^T = \ln\left(\frac{S_T}{S_t}\right)$, continousely compounded return, in "The Skewness Implied in the Heston Model and Its Application"
- unconditional skewness $=-\sigma\frac{\sqrt{2}A}{\sqrt{\kappa}B^{3/2}}$ with ![unconditional skewness, Zhang et al 2017](unconditional_skewness_zhang_2017.png)

Zhao et al 2013: Variance of $R_t^T = \mathbb{E}(R_t^T-\mathbb{E}(R_t^T))^2 = \frac{1}{4}\text{Var}(\int_t ^T V_sds) + SW + \mathbb{E}(\int_t^T \sqrt{V_s}dB_s(\int_t^T V_sdx-\mathbb{E}(\int_t^T V_sds)))$ with ![alt text](conditional_variance_zhao_2013.png), replace $V_t$ with $\theta$ as in (Zhang 2017) gives unconditional variance, alternatively (unconditional) variance is also included in Zhang 2017 paper

Moments of $Q_{t+1}=\frac{S_{t+1}}{S_t}$ in "Estimating Option Prices with Heston’s Stochastic Volatility Model" ($r$ drift, $k=\kappa$)
- $\mathbb{E}(Q_{t+1})=\mu_1=1+r$
- $\mathbb{E}(Q_{t+1}^2)=\mu_2=(r+1)^2+\theta$
- $\mathbb{E}(Q_{t+1}^3)=\mu_3=(r+1)^3+3\theta+3r\theta$
- $\mathbb{E}(Q_{t+1}^4)=\mu_4=\frac{1}{k(k-2)}(k^2r^4+4k^2r^3+6k^2r^2\theta+ \dots)$
![alt text](moments_dunn.png)

Moments of $S_t$ seam hard to find: "Unfortunately, a closed-form formula for the skewness has never been presented" (Zhang et al 2017)

both papers use Mathematica to get exact formulas, and suggest using characteristic function
- Wikipedia: If a random variable $X$ has moments up to $k$-th order, then the characteristic function $\phi_X$ is $k$ times continuously differentiable on the entire real line. In this case $\mathbb{E}(X^k)=i^{-k}\phi_X^{(k)}(0)$

---

Simulation of the process with discretisation gives realised moments (Haozhen works on that) and realised quantiles

---

Realised Moments + Expansion Method $\to$ pdf/cdf $\to$ compare to theoretical distribution
- which expansion method works best?

---

Comparing realised moments and theoretical moments was done by Neuberger & Payne (2021)
- with short term skew + kurtosis, long term skew and kurtosis are precisely estimatable