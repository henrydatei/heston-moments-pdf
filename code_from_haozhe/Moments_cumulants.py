import numpy as np
import math
from scipy.special import factorial

def MomentsCIR(p, q, kappa, theta, sigma, v0, t, conditional=True, comoments=False):
    if isinstance(p, (list, tuple, np.ndarray)):
        if not all(isinstance(x, int) for x in p) or any(x < 1 for x in p):
            raise ValueError("All elements of p must be integers >= 1")
    else:
        raise ValueError("p must be a list, tuple, or numpy array")
    
    if isinstance(q, (list, tuple, np.ndarray)):
        if not all(isinstance(x, int) for x in q) or any(x < 0 for x in q):
            raise ValueError("All elements of q must be integers >= 0")
    else:
        raise ValueError("q must be a list, tuple, or numpy array")

    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if v0 < 0:
        raise ValueError("v0 must be >= 0")
    if t <= 0:
        raise ValueError("t must be > 0")
    if not isinstance(conditional, bool):
        raise ValueError("conditional must be a logical")
    if not isinstance(comoments, bool):
        raise ValueError("comoments must be a logical")
    if comoments and q == [0]:
        raise ValueError("If comoments are TRUE, q is needed")
    if comoments == False and q != [0]:
        raise ValueError("If comoments are FALSE, q is not needed")

    def aFun(p, kappa, v0):
        return v0**(p - 1) / kappa

    def bFun(kappa, t):
        return math.exp(kappa * t) - 1

    def cFun(p, kappa, theta, sigma):
        return (p - 1) * (kappa * theta + 0.5 * (p - 2) * sigma**2)

    def IFun(k, kappa, t):
        return bFun(kappa, t)**k / (factorial(k) * kappa**(k - 1))

    def mFun(k, p, kappa, theta, sigma, v0):
        res = np.zeros((len(p), len(k)))
        for j in range(len(p)):
            for i in range(len(k)):
                if k[i] == 0:
                    res[j, i] = 0
                else:
                    res[j, i] = aFun(p[j] - k[i], kappa, v0) * np.prod([cFun(m, kappa, theta, sigma) for m in range(p[j] - k[i] + 1, p[j]+1)])
        return res

    def nuFun(p, kappa, theta, sigma, v0, t):
        mI = np.zeros(len(p))
        for i in range(len(p)):
            mI[i] = np.sum(mFun(np.arange(p[i]), p, kappa, theta, sigma, v0)[i, :p[i]] * IFun(np.arange(1, p[i] + 1), kappa, t))
        return aFun(p, kappa, v0) * bFun(kappa, t) + mI

    if conditional and not comoments:
        def Evp_v0(p, kappa, theta, sigma, v0, t):
            res = np.exp(-p * kappa * t) * (v0**p + cFun(p + 1, kappa, theta, sigma) * nuFun(p, kappa, theta, sigma, v0, t))
            return res

        return Evp_v0(np.array(p), kappa, theta, sigma, v0, t)

    elif conditional and comoments:
        def Evp_v0(p, kappa, theta, sigma, v0, t):
            res  = np.exp(-p * kappa * t) * (v0**p + cFun(p + 1, kappa, theta, sigma) * nuFun(p, kappa, theta, sigma, v0, t))
            return res

        def Evpvq_v0(p, q, kappa, theta, sigma, v0, t):
            res = np.zeros((len(p), len(q)))
            for i in range(len(p)):
                res[i] = v0**p[i] * Evp_v0(q, kappa, theta, sigma, v0, t)
            return res

        return Evpvq_v0(np.array(p), np.array(q), kappa, theta, sigma, v0, t)

    elif not conditional and not comoments:
        def Evp(p, kappa, theta, sigma, v0):
            res = np.zeros(len(p))
            for i in range(len(p)):
                if p[i] == 1:
                    res[i] = cFun(2, kappa, theta, sigma) * aFun(1, kappa, v0)
                else:
                    res[i] = cFun(p[i] + 1, kappa, theta, sigma) * mFun([p[i] - 1], [p[i]], kappa, theta, sigma, v0)[0, 0] / (math.factorial(p[i]) * kappa**(p[i] - 1))   # here take the only element of the m function since we synthetically transform the single scalar input to an one-element matrix
            return res

        return Evp(np.array(p), kappa, theta, sigma, v0)

    elif not conditional and comoments:
        def Evp(p, kappa, theta, sigma, v0):
            res = np.zeros(len(p))
            for i in range(len(p)):
                if p[i] == 1:
                    res[i] = cFun(2, kappa, theta, sigma) * aFun(1, kappa, v0)
                else:
                    res[i] = cFun(p[i] + 1, kappa, theta, sigma) * mFun([p[i] - 1], [p[i]], kappa, theta, sigma, v0)[0,0] / (math.factorial(max(p[i],1)) * float(kappa)**(p[i] - 1))
            return res

        def Evp_m(k, p, q, kappa, theta, sigma, v0, t):
            res = np.zeros((len(p), len(q), len(k)))
            for n in range(len(p)):
                for j in range(len(q)):
                    for i in range(len(k)):
                        if k[i] == 0:
                            res[n, j, i] = 0
                        else:
                            res[n, j, i] = Evp([p[n] + q[j] - k[i] - 1], kappa, theta, sigma, v0)[0] / kappa * np.prod([cFun(m, kappa, theta, sigma) for m in range(q[j] - k[i] + 1, q[j] + 1)])
            return res

        def Evp_nuq(p, q, kappa, theta, sigma, v0, t):
            vmI = np.zeros((len(p), len(q)))
            evp = np.zeros((len(p), len(q)))
            for n in range(len(p)):
                for j in range(len(q)):
                    vmI[n, j] = np.sum(Evp_m(np.arange(q[j]), p, q, kappa, theta, sigma, v0, t)[n, j,:q[j]] * IFun(np.arange(1, q[j] + 1), kappa, t))
                    evp[n, j] = Evp([p[n] + q[j] - 1], kappa, theta, sigma, v0)
            return evp * bFun(kappa, t) / kappa + vmI

        def Evpvq(p, q, kappa, theta, sigma, v0, t):
            res = np.zeros((len(p), len(q)))
            for i in range(len(p)):
                for j in range(len(q)):
                    res[i, j] = math.exp(-q[j] * kappa * t) * (Evp([p[i] + q[j]], kappa, theta, sigma, v0)[0] + cFun(q[j] + 1, kappa, theta, sigma) * Evp_nuq([p[i]], [q[j]], kappa, theta, sigma, v0, t)[0,0])
            return res

        return Evpvq(np.array(p), np.array(q), kappa, theta, sigma, v0, t)
    
def getSkFromMoments(m1, m2, m3):
    sk = (m3 - 3 * m2 * m1 + 2 * m1 ** 3) / (m2 - m1 ** 2) ** 1.5
    return sk

def getKuFromMoments(m1, m2, m3, m4):
    ku = (m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4) / (m2 - m1 ** 2) ** 2
    return ku


from scipy.special import comb

def MomentsBates(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0, conditional = True, nc=False):
    """
    This function calculates the first four moments of a Bates process

    Args:
        mu: Drift of the price process
        kappa: Rate of mean reversion.
        theta: Long-term mean.
        sigma: Volatility of the variance diffusion process.
        rho: Correlation between Brownian motions.
        lambdaj: Intensity parameter for jumps.
        muj: Mean of the jump size J which is log-normally distributed.
        vj: Volatility of the jump size J which is log-normally distributed.
        t: Time horizon.
        v0: Initial volatility.

    Returns:
        A dictionary containing mean, variance, skewness, and kurtosis.
    """

    # Input validation
    if not all([kappa > 0, theta > 0, sigma > 0, -1 <= rho <= 1, lambdaj >= 0, vj >= 0, t >= 0]):
        raise ValueError("Invalid parameters")

    # Define functions for each moment calculation 
    def Ert1_v0(mu, kappa, theta, t, v0):
        Ert1_v0 = (2 * t * kappa * mu + theta - t * kappa * theta - v0 + (-theta + v0) / np.exp(t * kappa)) / (2 * kappa)
        return Ert1_v0
    
    def Ert2_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):
        
        Ert2_v0 = ((2 * t**2 * np.exp(2 * t * kappa) * kappa**3 * (- 2 * mu + theta)**2 -
        (-1 + np.exp(t * kappa)) * (np.exp(t * kappa) * (-2 * kappa * (rho * sigma * (8 * theta - 4 * v0) +
        (theta - v0)**2) + sigma**2 * (5 * theta - 2 * v0) + 8 * kappa**2 * (theta - v0)) +
        sigma**2 * (theta - 2 * v0) + 2 * kappa * (theta - v0)**2) +
        2 * t * np.exp(t * kappa) * kappa * (2 * (sigma**2 + kappa * (-2 * (mu + rho * sigma) + theta)) * (theta - v0) +
        np.exp(t * kappa) *(sigma**2 * theta + kappa **2 * (4 * theta + lambdaj * vj * (4 + vj)) +
        2 * kappa * (2 * mu * (theta - v0) + theta * (-2 * rho * sigma - theta + v0)))) -
        8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj)) * np.log(1 + muj))) / (8 * np.exp(2 * t * kappa) * kappa**3)
        
        return Ert2_v0
    
    def Ert3_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):

        Ert3_v0 = (2 * t**3 * np.exp(3 * t * kappa) * kappa**5 * (2 * mu - theta)**3 + 
        6 * t**2 * np.exp(2 * t * kappa) * kappa**2 * ( - ((sigma**2 + kappa * (-2 * (mu + rho * sigma) + theta))**2 * (theta - v0)) +
        np.exp(t * kappa) * kappa * (2 * mu - theta) * (sigma**2 * theta + kappa**2 * (4 * theta + lambdaj * vj * (4 + vj)) + kappa * (2 * mu * (theta - v0) + theta * (-4 * rho * sigma - theta + v0)))) -   
        t * np.exp(t * kappa) * kappa * (-3 * (-2 * sigma**2 + kappa * (2 * mu + 4 * rho * sigma - theta)) * (sigma**2 * (theta - 2 * v0) + 2 * kappa * (theta - v0)**2) -   
        6 * np.exp(t * kappa) * (kappa**3 * (8 * mu + 8 * rho * sigma - 8 * theta - lambdaj * vj * (4 + vj)) * (theta - v0) +   
        sigma**4 * (-3 * theta + v0) + kappa * sigma**2 * (4 * mu * theta + 16 * rho * sigma * theta - theta**2 - 8 * rho * sigma * v0 - 3 * theta * v0 + 2 * v0**2) +
        2 * kappa**2 * (theta * (-4 * sigma * (sigma + 2 * rho * (mu + rho * sigma)) - 2 * (mu - 2 * rho * sigma) * theta + theta**2) +
        2 * (2 * sigma * (sigma + rho * (mu + rho * sigma)) + 2 * mu * theta - theta**2) * v0 + (-2 * (mu + rho * sigma) + theta) * v0**2)) +
        np.exp(2 * t * kappa) * (6 * sigma**4 * theta + 2* kappa**4 * lambdaj * vj**2 * (12 + vj) - 6 * kappa**2 * (2 * mu * (rho * sigma * (8 * theta - 4*v0) + (theta - v0)**2) -
        theta * ((4 + 8 * rho**2) * sigma**2 + 4 * rho * sigma * (3 * theta - 2 * v0) + (theta - v0)**2)) - 6 * kappa**3 * (8 * rho * sigma * theta + (8 * theta + lambdaj * vj * (4 + vj)) * (theta - v0) + 8 * mu * (-theta + v0)) -
        3 * kappa * sigma**2 * (theta * (12 * rho * sigma + 7 * theta - 4 * v0) + mu * (-10 * theta + 4 * v0)))) - (-1 + np.exp(t * kappa)) * (-(sigma**4 * (theta - 3 * v0)) - 3 * kappa * sigma**2 * (theta - 2 * v0) * (theta - v0) -   
        2 * kappa**2 * (theta - v0)**3 + np.exp(t * kappa) * (theta * (sigma**2 + 2 * kappa * theta) * (-12 * kappa**2 - 7 * sigma**2 + 2 * kappa * (12 * rho * sigma + theta)) + 
        3 * (sigma**2 + 2 * kappa * theta) * (8 * kappa**2 + 3 * sigma**2 - 2 * kappa * (6 * rho * sigma + theta)) * v0 +  12 * kappa**2 * (-2 * kappa + 2 * rho * sigma + theta) * v0**2 - 4 * kappa**2 * v0**3) +    
        np.exp(2 * t * kappa) * (24 * kappa**3 * (rho * sigma * (4 * theta - 2 * v0) + (theta - v0)**2) + 2 * sigma**4 * (-11 * theta + 3 * v0) + 3 * kappa * sigma**2 * (5 * theta * (8 * rho * sigma + theta) - (12 * rho * sigma + 7 * theta) * v0 + 2 * v0**2) +   
        2 * kappa**2 * (- (theta - v0)**3 + 6 * sigma**2 * (-5 * theta - 12 * rho**2 * theta + 2 * v0 + 4 * rho**2 * v0) - 12 * rho * sigma * (2 * theta**2 - 3 * theta * v0 + v0**2)))) - 
        12 * t * np.exp(2 * t * kappa) * kappa**4 * lambdaj * vj * (-2 * theta + np.exp(t * kappa) * (kappa * (-4 + 4 * t * mu - 2 * t * theta - vj) + 2*(theta - v0)) + 2 * v0) * np.log(1 + muj) + 
        24 * t * np.exp(2 * t * kappa) * kappa**4 * lambdaj * (-theta + v0 - np.exp(t * kappa) * (-theta + t * kappa * (-2 * mu + theta) + kappa * vj + v0)) * np.log(1 + muj)**2 + 
        16 * t * np.exp(3 * t * kappa) * kappa**5 * lambdaj * np.log(1 + muj)**3) / (16 * np.exp(3 * t * kappa) * kappa**5)
    
        return Ert3_v0
    
    def Ert4_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):

        Ert4_v0 = (3 * sigma**6 * (theta - 4 * v0) + 
        4 * kappa**3 * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) +  v0)**4 +
        8 * np.exp(3 * t * kappa) * sigma**2 * (-16 * t**2 * kappa**6 * rho**2 * (-3 + t * rho * sigma) * (theta - v0) + 3 * sigma**4 * (7 * theta + 2 * v0) + 
        2 * kappa**3 * sigma * ((-192 * (rho + rho**3) + 6 * t * (9 + 40 * rho**2) * sigma - 42 * t**2 * rho * sigma**2 + t**3 * sigma**3) * theta + 
        (48 * rho**3 - 18 * t * (1 + 4 * rho**2) * sigma + 24 * t**2 * rho * sigma**2 - t**3 * sigma**3) * v0) + 
        12 * kappa**4 * ((4 + 24 * rho**2 - 8 * t * rho * (4 + 3 * rho**2) * sigma + t**2 * (3 + 14 * rho**2) * sigma**2 - t**3 * rho * sigma**3) * theta +   
        (-8 * rho**2 + 8 * t * rho * (2 + rho**2) * sigma - t**2 * (3 + 8 * rho**2) * sigma**2 + t**3 * rho * sigma**3) * v0) + 
        6 * kappa**2 * sigma**2 * ((15 + 80 * rho**2 - 35 * t * rho * sigma + 2 * t**2 * sigma**2) * theta + (3 + t * sigma * (7 * rho - t * sigma)) * v0) +  
        24 * t * kappa**5 * ((2 + 8 * rho**2 - 4 * t * (rho + rho**3) * sigma + t**2 * rho**2 * sigma**2) * theta +  
        (-2 + rho * (4 * t * sigma + rho * (-4 + t * sigma * (2 * rho - t * sigma)))) * v0) + 3 * kappa * sigma**3 * (t * sigma * (9 * theta - v0) - 10 * rho * (6 * theta + v0))) +
        12 * np.exp(2 * t * kappa) * sigma**2 * (sigma**4 * (7 * theta - 4 * v0) + 8 * kappa**4 * (1 + 2 * t * rho * sigma * (-2 + t * rho * sigma)) * (theta - 2 * v0) +
        4 * kappa**2 * sigma**2 * ((6 + 20 * rho**2 - 14 * t * rho * sigma + t**2 * sigma**2) * theta - 2 * (3 + 12 * rho**2 - 10 * t * rho * sigma + t**2 * sigma**2) * v0) -
        8 * kappa**3 * sigma * ((-3 * t * sigma + 2 * rho * (4 + t * sigma * (-4 * rho + t * sigma))) * theta + 2 * (3 * t * sigma + 2 * rho * (-3 + t * sigma * (3 * rho - t * sigma))) * v0) +
        2 * kappa * sigma**3 * (t * sigma * (5 * theta - 6 * v0) + 4 * rho * (-6 * theta + 5 * v0))) + 24 * np.exp(t * kappa) * sigma**4 * (-2 * kappa**2 * (-1 + t * rho * sigma) * (theta - 3 * v0) +
        sigma**2 * (theta - 2 * v0) + kappa * sigma * (t * sigma * (theta - 3 * v0) + rho * (-4 * theta + 10 * v0))) + np.exp(4 * t * kappa) * (192 * t * kappa**5 * (1 + 4 * rho**2) * sigma**2 * theta +
        4 * t * kappa**7 * lambdaj * vj**2 * (48 + vj * (24 + vj)) + 12 * kappa * sigma**5 * (5 * t * sigma * theta + 8 * rho * (22 * theta - 5 * v0)) + 3 * sigma**6 * (-93 * theta + 20 * v0) +
        96 * kappa**3 * sigma**3 * ((3 * t * sigma + 4 * rho * (10 + 8 * rho**2 + 3 * t * rho * sigma)) * theta - 4 * rho * (3 + 2 * rho**2) * v0) -
        96 * kappa**2 * sigma**4 * ((11 + 50 * rho**2 + 5 * t * rho * sigma) * theta - 3 * (v0 + 4 * rho**2 * v0)) -
        96 * kappa**4 * sigma**2 * ((5 + 4 * rho * (6 * rho + t * (3 + 2 * rho**2) * sigma)) * theta - 2 * (v0 + 4 * rho**2 * v0))) -
        32 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj**2 * (12 + vj) * np.log(1 + muj) + 96 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj * (4 + vj) * np.log(1 + muj)**2 -
        128 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj * np.log(1 + muj)**3 + 64 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * np.log(1 + muj)**4 +
        12 * kappa**2 * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) + v0)**2 * (sigma**2 * (theta - 2 * v0) +
        np.exp(2 * t * kappa) * (2 * t * kappa**3 * (4 * theta + lambdaj * vj * (4 + vj)) - 8 * kappa**2 * (theta + t * rho * sigma * theta - v0) +
        sigma**2 * (-5 * theta + 2 * v0) + 2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0)) +
        4 * np.exp(t * kappa) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) + kappa * sigma * (t * sigma * (theta - v0) + 2 * rho * (-2 * theta + v0))) -
        8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj)) * np.log(1 + muj)) +
        3 * kappa * (sigma**2 * (theta - 2 * v0) + np.exp(2 * t * kappa) * (2 * t * kappa**3 * (4 * theta + lambdaj * vj * (4 + vj)) - 8 * kappa**2 * (theta + t * rho * sigma * theta - v0) +
        sigma**2 * (-5 * theta + 2 * v0) + 2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0)) + 4 * np.exp(t * kappa) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) +
        kappa * sigma * (t * sigma * (theta - v0) + 2 * rho * (-2 * theta + v0))) - 8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj)) * np.log(1 + muj))**2 -
        8 * kappa * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) + v0) * (sigma**4 * (theta - 3 * v0) + 3 * np.exp(2 * t * kappa) * sigma * ((-16 * kappa**3 * (2 + t * kappa) * rho +
        8 * kappa**2 * (2 + 2 * t * kappa + (6 + t * kappa * (4 + t * kappa)) * rho**2) * sigma - 8 * kappa * (2 + t * kappa)**2 * rho * sigma**2 +
        (5 + 2 * t * kappa * (3 + t * kappa)) * sigma**3) * theta + (16 * kappa**3 * (1 + t * kappa) * rho - 8 * kappa**2 * (2 * t * kappa + (2 + t * kappa * (2 + t * kappa)) * rho**2)*sigma +
        8 * t * kappa**2 * (2 + t * kappa) * rho * sigma**2 + (1 - 2 * t * kappa * (1 + t * kappa)) * sigma**3) * v0) + 6 * np.exp(t * kappa) * sigma**2 * (-2 * kappa**2 * (-1 + t * rho * sigma) * (theta - 2*v0) +
        sigma**2 * (theta - v0) + kappa * sigma * (t * sigma * (theta - 2 * v0) + rho * (-4 * theta + 6 * v0))) + 2 * np.exp(3 * t * kappa) * (-24 * t * kappa**4 * rho * sigma * theta + t * kappa**5 * lambdaj * vj**2 * (12 + vj) +
        sigma**4 * (-11 * theta + 3 * v0) + 3 * kappa * sigma**3 * (20 * rho * theta + t * sigma * theta - 6 * rho * v0) +
        12 * kappa**3 * sigma * (4 * rho * theta + t * sigma * theta + 2 * t * rho**2 * sigma * theta - 2 * rho * v0) - 6 * kappa**2 * sigma**2 * ((5 + 3 * rho * (4 * rho + t * sigma)) * theta - 2 * (v0 + 2 * rho**2 * v0))) -
        4 * t * np.exp(3 * t * kappa) * kappa**5 * lambdaj * np.log(1 + muj)* (3 * vj * (4 + vj) - 6 * vj * np.log(1 + muj) + 4 * np.log(1 +muj)**2))) / (64 * np.exp(4 * t * kappa) * kappa**7)
        
        return Ert4_v0
    
    def Varrt_v0(kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):

        Varrt_v0 = (sigma**2 * (theta - 2*v0) +
                    4 * np.exp(kappa * t) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) + kappa * sigma * (t * sigma * (theta - v0) + 2 * rho*(-2 * theta + v0))) + 
                    np.exp(2 * kappa * t) * (2 * t * kappa**3 * (4 * theta + lambdaj * vj * (4 + vj)) - 8 * kappa**2 * (theta + t * rho * sigma * theta - v0) + sigma**2 * (-5 * theta + 2 * v0) +
                    2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0) + 8 * t * kappa**3 * lambdaj * np.log(1 + muj) * (-vj + np.log(1 + muj)))) / (8 * np.exp(2 * kappa * t) * kappa**3)

        return Varrt_v0
    
    def Skewrt_v0(kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):
        Skewrt_v0 = -(np.sqrt(2)*(sigma**4*(theta - 3*v0) + 3*np.exp(2 * t * kappa)*sigma*((-16*kappa**3*(2 + t*kappa)*rho + 8*kappa**2*(2 + 2*t*kappa + (6 + t*kappa*(4 + t*kappa))*rho**2)*sigma - 
                    8*kappa*(2 + t*kappa)**2*rho*sigma**2 + (5 + 2*t*kappa*(3 + t*kappa))*sigma**3)*theta + (16*kappa**3*(1 + t*kappa)*rho - 8*kappa**2*(2*t*kappa + (2 + t*kappa*(2 + t*kappa))*rho**2)*sigma + 8*t*kappa**2*(2 + t*kappa)*rho*sigma**2 + 
                    (1 - 2*t*kappa*(1 + t*kappa))*sigma**3)*v0) + 6*np.exp(t * kappa)*sigma**2*(-2*kappa**2*(-1 + t*rho*sigma)*(theta - 2*v0) + sigma**2*(theta - v0) + kappa*sigma*(t*sigma*(theta - 2*v0) + rho*(-4*theta + 6*v0))) + 
                    2*np.exp(3 * t * kappa)*(-24*t*kappa**4*rho*sigma*theta + t*kappa**5*lambdaj*vj**2*(12 + vj) + sigma**4*(-11*theta + 3*v0) + 3*kappa*sigma**3*(20*rho*theta + t*sigma*theta - 6*rho*v0) + 
                    12*kappa**3*sigma*(4*rho*theta + t*sigma*theta + 2*t*rho**2*sigma*theta - 2*rho*v0) - 6*kappa**2*sigma**2*((5 + 3*rho*(4*rho + t*sigma))*theta - 2*(v0 + 2*rho**2*v0))) - 
                    4*t*np.exp(3 * t * kappa)*kappa**5*lambdaj*np.log(1 + muj)*(3*vj*(4 + vj) - 6*vj*np.log(1 + muj) + 4*np.log(1+muj)**2))) / (np.exp(3 * t * kappa)*kappa**5*((sigma**2*(theta - 2*v0) + 4*np.exp(t * kappa)*(sigma**2*theta - 2*kappa**2*(-1 + t*rho*sigma)*(theta - v0) + kappa*sigma*(t*sigma*(theta - v0) + 2*rho*(-2*theta + v0))) + 
                    np.exp(2 * t * kappa)*(2*t*kappa**3*(4*theta + lambdaj*vj*(4 + vj)) - 8*kappa**2*(theta + t*rho*sigma*theta - v0) + sigma**2*(-5*theta + 2*v0) + 2*kappa*sigma*(8*rho*theta + t*sigma*theta - 4*rho*v0) + 
                    8*t*kappa**3*lambdaj*np.log(1 + muj)*(-vj + np.log(1 + muj))))/(np.exp(2 * t * kappa)*kappa**3))**1.5)
        
        return Skewrt_v0
    
    def Kurtrt_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0):

        Kurtrt_v0 = (np.exp(4 * t * kappa) * (-12 * kappa**3 * (2 * t * kappa * mu + theta - t * kappa * theta - v0 + (-theta + v0) / np.exp(t * kappa))**4 + 
               (12 * kappa**2 * (2 * t * kappa * mu + theta - t * kappa * theta - v0 + (-theta + v0) / np.exp(t * kappa))**2  * (2 * t**2 * np.exp(2 * t * kappa)* kappa**3 * (- 2 * mu + theta)**2 -
               (-1 + np.exp(t * kappa))*(np.exp(t * kappa)* (-2 * kappa * (rho * sigma * (8 * theta - 4 * v0) + (theta - v0)**2) + sigma**2 * (5 * theta - 2 * v0) + 8 * kappa**2 * (theta - v0)) + sigma**2 * (theta - 2 * v0) + 2 * kappa * (theta - v0)**2) + 
               2 * t * np.exp(t * kappa) * kappa * (2 * (sigma**2 + kappa * (-2 * (mu + rho * sigma) + theta)) * (theta - v0) + np.exp(t * kappa) * (sigma**2 * theta + kappa**2 * (4 * theta + lambdaj * vj * (4 + vj)) + 
               2 * kappa * (2 * mu * (theta - v0) + theta * (-2 * rho * sigma - theta + v0)))) - 8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj)) * np.log(1 + muj)))/ np.exp(2 * t * kappa) - 
               (8 * kappa * (2 * t * kappa * mu + theta - t * kappa * theta - v0 + (-theta + v0) / np.exp(t * kappa))* (2 * t**3 * np.exp(3 * t * kappa) * kappa**5 * (2 * mu - theta)**3 + 
               6 * t**2 * np.exp(2 * t * kappa) * kappa**2 * (-((sigma**2 + kappa * (-2 * (mu + rho * sigma) + theta))**2 * (theta - v0)) + np.exp(t * kappa) * kappa * (2 * mu - theta) * (sigma**2 * theta + kappa**2 * (4 * theta + lambdaj * vj * (4 + vj)) + 
               kappa * (2 * mu * (theta - v0) + theta * (-4 * rho * sigma - theta + v0)))) - t * np.exp(t * kappa) * kappa * (-3 * (-2 * sigma**2 + kappa * (2 * mu + 4 * rho * sigma - theta)) * (sigma**2 * (theta - 2 * v0) + 2 * kappa * (theta - v0)**2) - 
               6 * np.exp(t * kappa) * (kappa**3 * (8 * mu + 8 * rho * sigma - 8 * theta - lambdaj * vj * (4 + vj)) * (theta - v0) + sigma**4 * (-3 * theta + v0) + kappa * sigma**2 * (4 * mu * theta - theta**2 + 8 * rho * sigma * (2 * theta - v0) - 
               3 * theta * v0 + 2 * v0**2) + 2 * kappa**2 * (theta * (-4 * sigma * (sigma + 2 * rho * (mu + rho * sigma)) - 2 * (mu - 2 * rho * sigma) * theta + theta**2) + 2 * (2 * sigma * (sigma + rho * (mu + rho * sigma)) + 2 * mu * theta - theta**2)*v0 + 
               (-2 * (mu + rho * sigma) + theta) * v0**2)) + np.exp(2 * t *kappa) * (6 * sigma**4 * theta + 2 * kappa**4 * lambdaj * vj**2 * (12 + vj) - 6 * kappa**2 * (2 * mu * (rho * sigma * (8 * theta - 4 * v0) + (theta- v0)**2) -
               theta * ((4 + 8 * rho**2) * sigma**2 + 4 * rho * sigma * (3 * theta - 2 * v0) + (theta - v0)**2)) - 6 * kappa**3 * (8 * rho * sigma * theta + (8 * theta + lambdaj * vj * (4 + vj)) * (theta - v0) + 8 * mu * (-theta + v0)) - 
               3 * kappa * sigma**2 * (theta * (12 * rho * sigma + 7 * theta - 4 * v0) + mu * (-10 * theta + 4 * v0)))) - (-1 + np.exp(t * kappa)) * (-(sigma**4 * (theta - 3 * v0)) - 3 * kappa * sigma**2 * (theta - 2 * v0) * (theta - v0) -
               2 * kappa**2 * (theta - v0)**3 + np.exp(t * kappa) * (theta * (sigma**2 + 2 * kappa * theta) * (-12 * kappa**2 - 7 * sigma**2 + 2 * kappa * (12 * rho * sigma + theta)) +3*(sigma**2 + 2 * kappa * theta) * (8 * kappa**2 + 3 * sigma**2 -  
               2 * kappa * (6 * rho * sigma + theta)) * v0 +  12 * kappa**2 * (-2 * kappa + 2 * rho * sigma + theta) * v0**2 - 4 * kappa**2 * v0**3) + np.exp(2 * t * kappa) * (24 * kappa**3 * (rho * sigma * (4 * theta - 2 * v0) + (theta - v0)**2) +
               2 * sigma**4 * (-11 * theta + 3 * v0) + 3 * kappa * sigma**2 * (5 * theta**2 + 4 * rho * sigma * (10 * theta - 3 * v0) - 7 * theta * v0 + 2 * v0**2) - 2 * kappa**2 * ((theta - v0)**3 +
               6 * sigma**2 * (5 * theta + 12 * rho**2 * theta - 2 * v0 - 4 * rho**2 * v0) + 12 * rho * sigma * (2 * theta**2 - 3 * theta * v0 + v0**2)))) - 
               12 * t * np.exp(2 * t * kappa) * kappa**4 * lambdaj * vj * (-2 * theta + np.exp(t * kappa) * (kappa * (-4 + 4 * t * mu - 2 * t * theta - vj) + 2 * (theta - v0)) + 2 * v0) * np.log(1 + muj) +
               24 * t * np.exp(2 * t * kappa) * kappa**4 * lambdaj * (-theta + v0 - np.exp(t * kappa) * (-theta + t * kappa * (-2 * mu + theta) + kappa * vj + v0))* np.log(1 + muj)**2 +
               16 * t * np.exp(3 * t * kappa) * kappa**5 * lambdaj * np.log(1 + muj)**3)) / np.exp(3 * t * kappa) + (3 * sigma**6 * (theta - 4 * v0) + 4 * kappa**3 * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) + v0)**4 + 
               12 * np.exp(2 * t * kappa) * sigma**2 * (sigma**4 * (7 * theta - 4 * v0) + 8 * kappa**4 * (1 + 2 * t * rho * sigma * (-2 + t * rho * sigma)) * (theta - 2*v0) + 2 * kappa * sigma**3 * (-24 * rho * theta + 5 * t * sigma * theta + 20 * rho * v0 - 6 * t * sigma * v0) +   
               4 * kappa**2 * sigma**2 * ((6 + 20 * rho**2 - 14 * t * rho * sigma + t**2 * sigma**2)*theta - 2 * (3 + 12 * rho**2 - 10 * t * rho * sigma + t**2 * sigma**2) * v0) - 
               8 * kappa**3 * sigma * ((-3 * t * sigma + 2 * rho * (4 + t * sigma * (-4 * rho + t * sigma))) * theta + 2 * (3 * t * sigma + 2 * rho * (-3 + t * sigma * (3 * rho - t * sigma))) * v0)) + 
               8 * np.exp(3 * t * kappa) * sigma**2 * (-16 * t**2 * kappa**6 * rho**2 * (-3 + t * rho * sigma) * (theta - v0) + 3 * sigma**4 * (7 * theta + 2 * v0) + 2 * kappa**3 * sigma * ((-192 * (rho + rho**3) + 6 * t * (9 + 40 * rho**2)*sigma - 
               42 * t**2 * rho * sigma**2 + t**3 * sigma**3) * theta + (48 * rho**3 - 18 * t * (1 + 4 * rho**2) * sigma + 24 * t**2 * rho * sigma**2 - t**3 * sigma**3) * v0) +
               12 * kappa**4 * ((4 + 24 * rho**2 - 8 * t * rho * (4 + 3 * rho**2) * sigma + t**2 * (3 + 14 * rho**2) * sigma**2 - t**3 * rho * sigma**3) * theta + (-8 * rho**2 +  
               8 * t * rho * (2 + rho**2) * sigma - t**2 * (3 + 8 * rho**2) * sigma**2 + t**3 * rho * sigma**3) * v0) + 6 * kappa**2 * sigma**2 * ((15 + 80 * rho**2 - 35 * t * rho * sigma + 2 * t**2 * sigma**2)*theta + 
               (3 + t * sigma * (7 * rho - t * sigma)) * v0) + 24 * t * kappa**5 * ((2 + 8 * rho**2 - 4 * t * (rho + rho**3) * sigma + t**2 * rho**2 * sigma**2) * theta + (-2 + rho * (4 * t * sigma + rho * (-4 + t * sigma * (2 * rho - t * sigma)))) * v0) + 
               3 * kappa * sigma**3 * (t * sigma * (9 * theta - v0) - 10 * rho * (6 * theta + v0))) + 24 * np.exp(t * kappa) * sigma**4 * (-2 * kappa**2 * (-1 + t * rho * sigma) * (theta - 3 * v0) +
               sigma**2 * (theta - 2 * v0) + kappa * sigma * (t * sigma * (theta - 3 * v0) + rho * (-4 * theta + 10 * v0))) +  np.exp(4 * t * kappa) * (192 * t * kappa**5 * (1 + 4 * rho**2) * sigma**2 * theta +
               4 * t * kappa**7 * lambdaj * vj**2 * (48 + vj * (24 + vj)) + 12 * kappa * sigma**5 * (5 * t * sigma * theta + 8 * rho * (22 * theta - 5 * v0)) + 3 * sigma**6 * (-93 * theta + 20 * v0) +  
               96 * kappa**3 * sigma**3 * ((3 * t * sigma + 4 * rho * (10 + 8 * rho**2 + 3 * t* rho * sigma)) * theta - 4 * rho * (3 + 2 * rho**2) * v0) - 96 * kappa**2 * sigma**4 * ((11 + 50 * rho**2 +
               5 * t * rho * sigma) * theta - 3 * (v0 + 4 * rho**2 * v0)) - 96 * kappa**4 * sigma**2 * ((5 + 4 * rho * (6 * rho + t * (3 + 2 * rho**2) * sigma)) * theta - 2 * (v0 + 4 * rho**2 * v0))) -
               32 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj**2 * (12 + vj) * np.log(1 + muj) + 96 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj * (4 + vj) * np.log(1 + muj)**2 - 
               128 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * vj * np.log(1 + muj)**3 + 64 * t * np.exp(4 * t * kappa) * kappa**7 * lambdaj * np.log(1+ muj)**4 +  
               12 * kappa**2 * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) + v0)**2 * (sigma**2 * (theta - 2 * v0) + np.exp(2 * t * kappa) * (2 * t* kappa**3 * (4 * theta + lambdaj * vj*(4 + vj)) - 
               8 * kappa**2 * (theta + t * rho * sigma * theta - v0) + sigma**2 * (-5 * theta + 2 * v0) + 2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0)) + 
               4 * np.exp(t * kappa) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) + kappa * sigma * (t * sigma * (theta - v0) + 2 * rho * (-2 * theta + v0))) -  
               8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj)) * np.log(1 + muj)) + 3 * kappa * (sigma**2 * (theta - 2 * v0) + np.exp(2 * t * kappa) * (2 * t * kappa**3 * (4 * theta + lambdaj * vj * (4 + vj)) - 
               8 * kappa**2 * (theta + t * rho * sigma * theta - v0) + sigma**2 * (-5 * theta + 2 * v0) + 2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0)) +   
               4 * np.exp(t * kappa) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) + kappa * sigma * (t * sigma * (theta - v0) + 2 * rho * (-2 * theta + v0))) - 
               8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * (vj - np.log(1 + muj))*np.log(1 + muj))**2  - 8 * kappa * (-theta + np.exp(t * kappa) * (t * kappa * (2 * mu - theta) + theta - v0) + v0) *  
               (sigma**4 * (theta - 3 * v0) + 3 * np.exp(2 * t * kappa) * sigma * ((-16 * kappa**3 * (2 + t * kappa) * rho + 8 * kappa**2 * (2 + 2 * t * kappa + (6 + t * kappa * (4 + t * kappa)) * rho**2)*sigma -  
               8 * kappa * (2 + t * kappa)**2 * rho * sigma**2 + (5 + 2 * t * kappa * (3 + t * kappa)) * sigma**3) * theta + (16 * kappa**3 * (1 + t * kappa) * rho -  
               8 * kappa**2 * (2 * t * kappa + (2 + t * kappa * (2 + t * kappa)) * rho**2)*sigma + 8 * t * kappa**2 * (2 + t * kappa) * rho * sigma**2 + (1 - 2 * t * kappa * (1 + t * kappa)) * sigma**3) * v0) +
               6 * np.exp(t * kappa) * sigma**2 * (-2 * kappa**2 * (-1 + t * rho * sigma) * (theta - 2 * v0) + sigma**2 * (theta - v0) + kappa * sigma * (t * sigma * (theta - 2 * v0) + rho * (-4 * theta + 6 * v0))) +
               2 * np.exp(3 * t * kappa) * (-24 * t * kappa**4 * rho * sigma * theta + t * kappa**5 * lambdaj * vj**2 * (12 + vj) + sigma**4 * (-11 * theta + 3 * v0) + 3 * kappa * sigma**3 * (20 * rho * theta + t * sigma * theta - 6 * rho * v0) + 
               12 * kappa**3 * sigma * (4 * rho * theta + t * sigma * theta + 2 * t * rho**2 * sigma * theta - 2 * rho * v0) - 6 * kappa**2 * sigma**2 *((5 + 3 * rho * (4 * rho + t * sigma)) * theta - 2 * (v0 + 2 * rho**2 * v0))) -  
               4 * t * np.exp(3 * t * kappa) * kappa**5 * lambdaj * np.log(1 + muj) * (3 * vj * (4 + vj) - 6 * vj * np.log(1 + muj) + 4 * np.log(1 + muj)**2))) / np.exp(4 * t * kappa))) / (kappa * (sigma**2 * (theta - 2 * v0) + np.exp(2 * t * kappa) * (2 * t * kappa**3 * (4 * theta + lambdaj * vj * (4 + vj)) - 8 * kappa**2 * (theta + t * rho * sigma * theta - v0) +  
               sigma**2 * (-5 * theta + 2 * v0) + 2 * kappa * sigma * (8 * rho * theta + t * sigma * theta - 4 * rho * v0)) + 4 * np.exp(t * kappa) * (sigma**2 * theta - 2 * kappa**2 * (-1 + t * rho * sigma) * (theta - v0) +
               kappa * sigma * (-4 * rho * theta + t * sigma * theta + 2 * rho * v0 - t * sigma * v0)) - 8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * vj * np.log(1 + muj) +  8 * t * np.exp(2 * t * kappa) * kappa**3 * lambdaj * np.log(1 + muj)**2)**2)

        return Kurtrt_v0
    

    def Ert1(mu, theta, t):
        Ert1 = (mu - theta/2)*t
        return Ert1
    
    def Ert2(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t):

        Ert2 = (sigma * (- 4 * kappa * rho + sigma) * theta + np.exp(kappa * t) * (- (sigma**2 * theta) - 4 * kappa**2 * rho * sigma * t * theta + kappa * sigma * (4 * rho + sigma * t) * theta +
        kappa**3 * t * (4 * theta + t * (-2 * mu + theta)**2 + lambdaj * vj * (4 + vj)) + 4 * kappa**3 * lambdaj * t * np.log(1 + muj) * (-vj + np.log(1 + muj)))) / (4 * np.exp(kappa * t) * kappa**3)
 
        return Ert2
    
    def Ert3(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t):

        Ert3 =  (- 3 * sigma * theta * (2 * sigma**3 + kappa * sigma**2 * (- 12 * rho + sigma * t) + 4 * kappa**3 * rho*(- 2 + 2 * mu * t + 2 * rho * sigma * t - t * theta) + 
                kappa**2 * sigma * (4 + 16 * rho**2 - 2 * mu * t - 6 * rho * sigma * t + t * theta)) + 
                np.exp(kappa * t) * (6 * sigma**4 * theta - 3 * kappa * sigma**3 * (12 * rho + sigma * t) * theta + 12 * kappa**4 * rho * sigma * t * theta * (2 - 2 * mu * t + t * theta) + 
                3 * kappa**2 * sigma**2 * theta * (4 + 16 * rho**2 - 2 * mu * t + 6 * rho * sigma * t + t * theta) - 
                3 * kappa**3 * sigma * theta * (8 * rho**2 * sigma * t + sigma * t * (4 - 2 * mu * t + t * theta) + rho * (8 - 8 * mu * t + 4 * t * theta)) + 
                kappa**5 * t * (t * (2 * mu - theta) * (12 * theta + t * (- 2 * mu + theta)**2) + 12 * lambdaj * t * (2 * mu - theta) * vj + 3 * lambdaj * (- 4 + 2 * mu * t - t * theta) * vj**2 - 
                lambdaj * vj**3)) + 2 * np.exp(kappa * t) * kappa**5 * lambdaj * t * np.log(1 + muj) * (3 * vj * (4 - 4 * mu * t + 2 * t * theta + vj) - 6 * (-2 * mu * t + t * theta + vj) * np.log(1 + muj) + 4 * np.log(1 + muj)**2)) / (8 * np.exp(kappa * t) * kappa**5)
    
        return Ert3
    
    def Ert4(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t):

        Ert4 = (( - 24 * kappa**2 * (2 * kappa * rho - sigma) * sigma * t * (2 * (- 1 + np.exp(kappa * t)) * sigma**2 - kappa * sigma * (8 * (- 1 + np.exp(kappa * t)) * rho + sigma * t) 
           + 4 * kappa**2 * (- 1 + np.exp(kappa * t) + rho * sigma * t)) * (2 * mu - theta) * theta)/np.exp(kappa * t) + (3 * sigma**2 * theta * ((-4 * kappa*rho + sigma)**2 * 
           (sigma**2 + 2 * kappa * theta) + np.exp(2 * kappa * t) * (- 32 * kappa**4 * (1 + 8 * rho**2) - 29 * sigma**4 + 2 * kappa * sigma**2 * (116 * rho * sigma + theta) - 
           16 * kappa**2 * sigma * (6 * sigma + 35 * rho**2 * sigma + rho * theta) + 32 * kappa**3 * rho * (12 * (1 + rho**2) * sigma + rho * theta)) - 
           4 * np.exp(kappa * t) * (- 7 * sigma**4 + 16 * kappa**5 * rho**2 * t * (-2 + rho * sigma * t) + 4 * kappa**4 * (- 2 + rho * (12 * sigma*t +
           rho * (- 16 + sigma * t * (16 * rho - 5 * sigma * t)))) + kappa * sigma**2 * (56 * rho * sigma - 5 * sigma**2 * t + theta) - 
           kappa**2 * sigma * (sigma * (24 + 136 * rho**2 - 40 * rho * sigma * t + sigma**2 * t**2) + 8 * rho * theta) + 
           4 * kappa**3 * (sigma * (24 * (rho + rho**3) - 3 * (1 + 8 * rho**2) * sigma * t + 2 * rho * sigma**2 * t**2) + 4 * rho**2 * theta)))) / np.exp(2 * kappa * t) + (12 * (- 1 + np.exp(kappa * t)) * kappa**2 * (4 * kappa * rho - sigma) * sigma * t * theta * (- 4 * kappa * rho * sigma * theta + 
           sigma**2 * theta + kappa**2 * (4 * theta + t * (- 2 * mu + theta)**2 + lambdaj * vj * (4 + vj)) + 4 * kappa**2 * lambdaj * np.log(1 + muj) * (- vj + np.log(1 + muj)))) / np.exp(kappa * t) + 
           2 * kappa * t * (- 120 * kappa * rho * sigma**5 * theta + 15 * sigma**6 * theta - 48 * kappa**3 * rho * sigma**3 * theta * (6 + 4 * rho**2 - 3 * mu * t + 2 * t * theta) + 
           3 * kappa**2 * sigma**4 * theta * (24 + 96 * rho**2 - 8 * mu * t + 5 * t * theta) + kappa**6 * (t * (16 * mu**4 * t**2 - 32 * mu**2 * t * (- 3 + mu * t) * theta +
           24 * (2 + mu * t * (- 4 + mu * t)) * theta**2 - 8 * t * (- 3 + mu * t) * theta**3 + t**2 * theta**4) + 24 * lambdaj * t * (4 * theta + t * (- 2 * mu + theta)**2) * vj + 
           6 * lambdaj * (8 + t * (8 * lambdaj + 4 * mu**2 * t - 4 * mu*(4 + t * theta) + theta * (12 + t * theta))) * vj**2 + 4 * lambdaj * (6 + t * (6 * lambdaj - 2 * mu + theta)) * vj**3 + 
           lambdaj * (1 + 3 * lambdaj * t) * vj**4) - 24 * kappa**5 * rho * sigma * t * theta * (4 * mu**2 * t - 4 * mu * (2 + t * theta) + theta * (8 + t * theta) + lambdaj * vj * (4 + vj)) + 
           6 * kappa**4 * sigma**2 * theta * (8 - 16 * mu * t + 8 * rho**2 * (4 - 4 * mu * t + 3 * t * theta) + t * (12 * theta + t * (- 2 * mu + theta)**2 + lambdaj * vj * (4 + vj))) + 
           8 * kappa**4 * lambdaj * np.log(1 + muj) * (vj * (12 * kappa * rho * sigma * t * theta - 3 * sigma**2 * t * theta + 
           kappa**2 * (- 12 * mu**2 * t**2 - 3 * t * theta * (8 + t * theta) - 3 * (4 + 4 * lambdaj * t + t * theta) * vj - (1 + 3 * lambdaj * t) * vj**2 + 
           6 * mu * t*(4 + 2 * t * theta + vj))) + np.log(1 + muj) * (3 * (- 4 * kappa * rho * sigma * t * theta + sigma**2 * t * theta + 
           kappa**2 * (t * (4 * theta + t * (-2 * mu + theta)**2) + 2 * (2 + t * (2 * lambdaj - 2 * mu + theta)) * vj + (1 + 3 * lambdaj * t) * vj**2)) + 
           2 * kappa**2 * np.log(1 + muj) * (- 2 * (vj + t * (- 2 * mu + theta + 3 * lambdaj * vj)) + (1 + 3 * lambdaj * t) * np.log(1 + muj)))))) / (32 * kappa**7)
      
        return Ert4
    
    def Skewrt(kappa, theta, sigma, rho, lambdaj, muj, vj, t):

        Skewrt = (- 3 * (2 * kappa * rho - sigma) * sigma * (- 2 * sigma**2 + kappa * sigma * (8 * rho - sigma * t) + 4 * kappa**2 * (- 1 + rho * sigma * t)) * theta - 
             np.exp(kappa * t) * (- 3 * (2 * kappa * rho - sigma) * sigma * (- 2 * sigma**2 + 4 * kappa**3 * t + kappa * sigma * (8 * rho + sigma * t) - 
             4 * kappa**2 * (1 + rho * sigma * t)) * theta + 12 * kappa**5 * lambdaj * t * vj**2 + kappa**5 * lambdaj * t * vj**3) + 
             2 * np.exp(kappa * t) * kappa**5 * lambdaj * t * np.log(1 + muj) * (3 * vj * (4 + vj) - 6 * vj * np.log(1 + muj) + 4 * np.log(1 + muj)**2)) / (np.exp(kappa * t) * kappa**5 * (((- 1 + np.exp(- kappa * t)) * sigma**2 * theta - 4 * kappa**2 * rho * sigma * t * theta + 
             kappa * sigma * ((4 - 4 / np.exp(kappa * t)) * rho + sigma * t) * theta + kappa**3 * t * (4 * theta + lambdaj * vj * (4 + vj))) / kappa**3 - 
             4 * lambdaj * t * vj * np.log(1 + muj) + 4 * lambdaj * t * np.log(1 + muj)**2)**1.5)
        
        return Skewrt
    
    def Kurtrt(kappa, theta, sigma, rho, lambdaj, muj, vj, t):

        Kurtrt = (3 * sigma**2 * (- 4 * kappa * rho + sigma)**2 * theta * (sigma**2 + 2 * kappa * theta) + 12 * np.exp(kappa * t) * sigma * theta * (7 * sigma**5 - 
            kappa * sigma**3 * (56 * rho*sigma - 5 * sigma**2 * t + theta) + kappa**2 * sigma**2 * (- 40 * rho * sigma**2 * t + sigma**3 * t**2 + 
            8 * rho * theta + sigma * (24 + 136 * rho**2 + t * theta)) - 4 * kappa**3 * sigma * (24 * rho**3 * sigma - 3 * sigma**2 * t + 4 * rho**2 * (- 6 * sigma**2 * t + theta) + 
            2 * rho * sigma * (12 + sigma**2 * t**2 + t * theta)) - 4 * kappa**5 * rho * t * (- 8 * rho * sigma + 4 * rho**2 * sigma**2 * t + 4 * theta + lambdaj * vj * (4 + vj)) + 
            kappa**4 * sigma * (8 - 48 * rho * sigma * t - 64 * rho**3 * sigma * t + 4 * rho**2 * (16 + 5 * sigma**2 * t**2 + 4 * t * theta) + t * (4 * theta + lambdaj * vj * (4 + vj)))) + 
            np.exp(2 * kappa * t) * (- 87 * sigma**6 * theta + 6 * kappa * sigma**4 * theta * (116 * rho * sigma + 5 * sigma**2 * t + theta) + 6 * kappa**3 * sigma**2 * theta * (192 * rho**3 * sigma + 
            16 * rho**2 * (6 * sigma**2 * t + theta) + 16 * rho * sigma * (12 + t * theta) + sigma**2 * t * (24 + t*theta)) - 12 * kappa**2 * sigma**3 * theta * (20 * rho * sigma**2 * t + 
            4 * rho * theta + sigma * (24 + 140 * rho**2 + t * theta)) - 48 * kappa**6 * rho * sigma * t**2 * theta * (4 * theta + lambdaj * vj * (4 + vj)) - 
            12 * kappa**4 * sigma**2 * theta * (8 + 32 * rho**3 * sigma * t + 16 * rho**2 * (4 + t * theta) + 4 * rho * sigma * t * (12 + t * theta) + 
            t * (4 * theta + lambdaj * vj * (4 + vj))) + 2 * kappa**7 * t * (lambdaj * vj**2 * (48 + 24 * vj + vj**2) + 3 * t * (4 * theta + lambdaj * vj * (4 + vj))**2) + 
            12 * kappa**5 * sigma * t * theta * (4 * rho * (4 * theta + lambdaj * vj * (4 + vj)) + sigma * (8 + 8 * rho**2 * (4 + t * theta) + t * (4 * theta + lambdaj * vj * (4 + vj))))) - 
            16 * np.exp(kappa * t) * kappa**4 * lambdaj * t * vj * (- 3 * (- 1 + np.exp(kappa * t)) * sigma**2 * theta - 12 * np.exp(kappa * t) * kappa**2 * rho * sigma * t * theta + 
            3 * kappa * sigma * (4 * (- 1 + np.exp(kappa * t)) * rho + np.exp(kappa * t) * sigma * t) * theta + np.exp(kappa * t) * kappa**3 * (vj * (12 + vj) +
            3 * t * (4 * theta + lambdaj * vj * (4 + vj)))) * np.log(1 + muj) + 48 * np.exp(kappa * t) * kappa**4 * lambdaj * t * ((1 - np.exp(kappa * t)) * sigma**2 * theta - 
            4 * np.exp(kappa * t) * kappa**2 * rho * sigma * t * theta + kappa * sigma * (4 * (- 1 + np.exp(kappa * t)) * rho + np.exp(kappa * t) * sigma * t) * theta + 
            np.exp(kappa * t) * kappa**3 * (vj * (4 + vj) + t * (4 * theta + lambdaj * vj * (4 + 3 * vj)))) * np.log(1 + muj)**2 - 
            64 * np.exp(2 * kappa * t) * kappa**7 * lambdaj * t * (1 + 3 * lambdaj * t) * vj * np.log(1 + muj)**3 + 32 * np.exp(2 * kappa * t) * kappa**7 * lambdaj * t * (1 + 3 * lambdaj * t) * np.log(1 + muj)**4) / (2 * np.exp(2 * kappa * t) * kappa**7 * (((- 1 + np.exp(- kappa * t)) * sigma**2 * theta - 4 * kappa**2 * rho * sigma * t * theta + 
            kappa * sigma * ((4 - 4 / np.exp(kappa * t)) * rho + sigma * t) * theta + kappa**3 * t * (4 * theta + lambdaj * vj * (4 + vj))) / kappa**3 - 4 * lambdaj * t * vj * np.log(1 + muj) + 
            4 * lambdaj * t * np.log(1 + muj)**2)**2)
      
        return Kurtrt

    # Moment calculations 
    if conditional:

        mean = Ert1_v0(mu, kappa, theta, t, v0)

        M2_v0 = Ert2_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)
        variance = Varrt_v0(kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)

        M3_v0 = Ert3_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)
        skewness = Skewrt_v0(kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)
        cumulant3 = M3_v0 - 3 * mean * variance - mean**3

        M4_v0 = Ert4_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)
        kurtosis = Kurtrt_v0(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t, v0)
        cumulant4 = M4_v0 - 4 * mean * M3_v0 + 6 * mean**2 * M2_v0 - 3 * mean**4

        if nc:

            return {"m1": mean, "m2": M2_v0, "m3": M3_v0, "m4": M4_v0}
        
        else:

            return {"mean": mean, "variance": variance, "skewness": skewness, "kurtosis": kurtosis, "3rd cumulant": cumulant3, "4th cumulant": cumulant4}

        #return {"mean": mean, "variance": (M2_v0 - mean**2), "skewness": getSkFromMoments(mean, M2_v0, M3_v0), "kurtosis": getKuFromMoments(mean, M2_v0, M3_v0, M4_v0)}

    else:

        mean = Ert1(mu, theta, t)

        M2_unc = Ert2(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t)
        variance = Ert2(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t) - mean**2

        M3_unc = Ert3(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t)
        skewness = Skewrt(kappa, theta, sigma, rho, lambdaj, muj, vj, t)
        cumulant3 = M3_unc - 3 * mean * variance - mean**3

        M4_unc = Ert4(mu, kappa, theta, sigma, rho, lambdaj, muj, vj, t)
        kurtosis = Kurtrt(kappa, theta, sigma, rho, lambdaj, muj, vj, t)
        cumulant4 = M4_unc - 4 * mean * M3_unc + 6 * mean**2 * M2_unc - 3 * mean**4

        if nc:

            return {"m1": mean, "m2": M2_unc, "m3": M3_unc, "m4": M4_unc}
        
        else:

            return {"mean": mean, "variance": variance, "skewness": skewness, "kurtosis": kurtosis, "3rd cumulant": cumulant3, "4th cumulant": cumulant4}

        #return {"mean": mean, "variance": variance, "skewness": getSkFromMoments(mean, M2_unc, M3_unc), "kurtosis": getKuFromMoments(mean, M2_unc, M3_unc, M4_unc)}
