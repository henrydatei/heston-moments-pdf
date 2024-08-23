# Implementation of moments from Neuberger (2012) and Bae & Lee (2021) and Fukasawa & Matsushita (2021)
# r is array of log returns
# s is array of log prices

import pandas as pd
import numpy as np

def g_M(delta_s: float) -> float:
    return np.exp(delta_s) - 1

def g_V(delta_s: float) -> float:
    return 2 * (np.exp(delta_s) - 1 - delta_s)

def realized_variance(s: pd.Series) -> float:
    '''not annualized'''
    return s.diff().dropna().apply(g_V).sum()

def implied_variance(s: pd.Series) -> float:
    first_price = s.iloc[0]
    last_price = s.iloc[-1]
    return g_V(last_price - first_price)

def g_Q(delta_s: float, delta_v_E: float) -> float:
    def K(delta_s: float) -> float:
        return 6 * (delta_s * np.exp(delta_s) - 2*np.exp(delta_s) + delta_s + 2)
    return 3 * delta_v_E * (np.exp(delta_s) - 1) + K(delta_s)

def L(x: float) -> float:
    return 2 * (np.exp(x) - 1 - x)

def E(x: float) -> float:
    return 2 * (x * np.exp(x) - np.exp(x) + 1)

def v_L(s: pd.Series) -> pd.Series:
    return s.diff().dropna().apply(L)

def v_E(s: pd.Series) -> pd.Series:
    return s.diff().dropna().apply(E)

def implied_third_moment(s: pd.Series) -> float:
    return 3 * (v_E(s) - v_L(s))

def implied_skew_coefficient(s: pd.Series) -> float:
    return implied_third_moment(s) / implied_variance(s)**1.5

def realized_third_moment(s: pd.Series) -> float:
    delta_ss = s.diff().dropna()
    delta_v_Es = v_E(s).diff().dropna()
    sum = 0
    for delta_s, delta_v_E in zip(delta_ss, delta_v_Es):
        sum += g_Q(delta_s, delta_v_E)
    return sum

def realized_skew_coefficient(s: pd.Series) -> float:
    return realized_third_moment(s) / realized_variance(s)**1.5