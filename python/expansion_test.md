# Expansion Method Tests
How good does the Gram-Charlier, GC with positivity constraints and Cornish-Fisher Expansion work for known distributions with known skewness and kurtosis?

Three distributions:
- standard normal
    - skewness 0
    - excess kurtosis 0
- lognormal, parameter: $\sigma^2$
    - skewness $(\exp(\sigma^2)+2)\cdot \sqrt{\exp(\sigma^2)-1}$
    - excess kurtosis $1\cdot\exp(4\sigma^2)+2\exp(3\sigma^2)+3\exp(2\sigma^2)-6$
- student's t, parameter: $\nu$
    - skewness 0 for $\nu>3$
    - excess kurtosis $\frac{6}{\nu-4}$ for $\nu>4$

Set parameters to $\sigma=0.5$ and $\nu=5$, yields (skewness, excess kurtosis):
- normal distribution 0 0
- lognormal: 1.7501896550697178 5.898445673784778
- t: 0 6

## Gram-Charlier Expansion

![alt text](gc_expansion.png)
- fit is not good
- gram-charlie expansion gets negative, not a density

## Gram-Charlier Expansion with Positivity Constraints

Transforms skewness and kurtosis to
- lognormal: 0.10234746774707859 3.989055204819389
- t: 0.0 3.9901095073734614

![alt text](gc_positivity_expansion.png)
- fit is not good
- alt least no negativity

![alt text](gc_positivity_boundary.png)

## Cornish-Fisher Expansion

![alt text](cf_expansion.png)
- fit is not good
- skewness has wrong sign (?), GC expansion seems to suffer from the same problem, middle of distribution is on the wrong side