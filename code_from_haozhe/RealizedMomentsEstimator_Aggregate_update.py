# Define constants for the realized moments
RM_ACJV = 1  # Amaya et al, 2015
RM_CL = 2    # Choe and Lee, 2014
RM_NP = 3    # Neuberger and Payne, 2018
RM_ORS = 4   # Okhrin, Rockinger and Schmid, 2020
RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

# Function to get the name of the realized moments
def getname_RM(x):
    if x == RM_ACJV:
        return "RM_ACJV"
    elif x == RM_CL:
        return "RM_CL"
    elif x == RM_NP:
        return "RM_NP"
    elif x == RM_NP_return:
        return "RM_NP_return"
    elif x == RM_ORS:
        return "RM_ORS"
    else:
        return "RM model you want is not implemented"


import numpy as np
import pandas as pd

# Function to compute skewness and kurtosis from moments
def moms2skkurt(moms, m1zero=True):
    set_index = moms
    if not isinstance(moms, np.ndarray):
        moms = np.array(moms).reshape(-1, 4)          # -1 means the unknown dimension which will be figured out by np
    if m1zero:
        real_mean = moms[:, 0]
        center = 0
    else:
        real_mean = moms[:, 0]
        center = real_mean
    VaT = moms[:, 1] - center**2
    SkT = (moms[:, 2] - 3 * moms[:, 1] * center + 2 * center**3) / ((moms[:, 1] - center**2)**(3/2))
    KuT = abs((moms[:, 3] - 4 * moms[:, 2] * center + 6 * moms[:, 1] * (center**2) - 3 * (center**4)) / ((moms[:, 1] - center**2)**2))
    #return np.column_stack((real_mean, VaT, SkT, KuT))
    outcome = np.column_stack((real_mean, VaT, SkT, KuT))
    outcome = pd.DataFrame(outcome, columns=['rMean', 'rVar', 'rSkew', 'rKurt'], index=[set_index.index])
    return outcome

def cumus2skkurt(cumus):
    set_index = cumus
    if not isinstance(cumus, np.ndarray):
        cumus = np.array(cumus).reshape(-1, 4) # -1 means the unknown dimension which will be figured out by np

    Mean = cumus[:, 0]
    VaT = cumus[:, 1] 
    SkT = cumus[:, 2] / ((cumus[:, 1])**(3/2))
    KuT = cumus[:, 3] / ((cumus[:, 1])**2)
    #return np.column_stack((Mean, VaT, SkT, KuT))
    outcome = np.column_stack((Mean, VaT, SkT, KuT))
    outcome = pd.DataFrame(outcome, columns=['rMean', 'rVar', 'rSkew', 'rKurt'], index=[set_index.index])
    return outcome

# Function to compute realized moments as in Amaya et al (2015) for one interval
def rMomACJV_l(hf: pd.Series):
    m1 = hf.resample('D').sum(min_count=1).dropna()  # Daily sum of hf
    m2 = (hf**2).resample('D').sum(min_count=1).dropna()  # Daily sum of hf^2
    m3 = (hf**3).resample('D').sum(min_count=1).dropna()  # Daily sum of hf^3
    m4 = (hf**4).resample('D').sum(min_count=1).dropna()  # Daily sum of hf^4
    return pd.DataFrame({'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4})

# Function to compute realized moments as in Amaya et al (2015) averaged over nD days
def rMomACJV(hf: pd.Series, nD: int):
    rolling_sums = rMomACJV_l(hf).rolling(window=nD).sum()
    avg_over_days = rolling_sums.dropna().mean()
    return pd.DataFrame([avg_over_days], index=[hf.index[-1]])

# Function to compute realized moments as in Choe and Lee (2014) for one interval
def rMomCL_l(hf: pd.Series):
    R = hf.cumsum()
    diff_R = R.diff().dropna()
    diff_R_sq = (R**2).diff().dropna()
    return np.array([0, (diff_R**2).sum(), 1.5 * np.sum(diff_R * diff_R_sq), 1.5 * (diff_R_sq**2).sum()]) 

# Function to compute realized moments as in Choe and Lee (2014) averaged over nD days
def rMomCL(hf: pd.Series, nD: int):
    daily_moments = hf.resample('D').apply(rMomCL_l)
    # remove weekends
    daily_moments = daily_moments[daily_moments.index.weekday < 5]
    df_expanded = pd.DataFrame(daily_moments.tolist(), index=daily_moments.index, columns=['m1', 'm2', 'm3', 'm4'])
    # print(df_expanded)
    rolling_sums = df_expanded.rolling(window=nD).sum()
    avg_over_days = rolling_sums.dropna().mean()
    return pd.DataFrame([avg_over_days], index=[hf.index[-1]])

# Function to compute realized moments Neuberger and Payne (2018)
def endpoints(df: pd.Series, freq: str):
    return df.groupby(df.index.to_period(freq)).apply(lambda x: x.index[-1])                # df.resample(freq).apply(lambda x: x.index[-1])

def startpoints(df: pd.Series, freq):
    return df.groupby(df.index.to_period(freq)).apply(lambda x: x.index[0]) 

def rMomNP(hf: pd.Series, nD: int):
    ep = endpoints(hf, 'D')  # define the endpoints of the data with respect to selected frequency
    ep = ep.to_numpy()  # Convert index to numpy array

    sp = startpoints(hf, 'D')  # define the start points of the data with respect to selected frequency
    sp = sp.to_numpy()  # Convert index to numpy array
    
    ints = np.column_stack((sp[:-nD*2+1], ep[nD*2-1:]))   # pairs the start and end indices of the intervals for the given number of days, i.e., (0,4), (1,5),..., (-4,:), here we take two aggregation intervals, i.e., the first one is only used to compute average (lagged)
    
    def apply_realized_estimators_NP(x):
        start, end = ints[x]
        data = hf[start:end]#.cumsum()
        return RealizedEstimatorsORS(data, nD, NP=True)
    
    #result = np.array([apply_realized_estimators_NP(x) for x in range(len(ints))])
    result = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4'])
    for x in range(len(ints)):

        result = pd.concat([result if not result.empty else None, apply_realized_estimators_NP(x)])
    
    return result

# Function to compute realized moments ORS (2020)
def rMomORS(hf, nD):
    ep = endpoints(hf, 'D')  # define the endpoints of the data with respect to selected frequency
    ep = ep.to_numpy()  # Convert index to numpy array
    
    sp = startpoints(hf, 'D')  # define the start points of the data with respect to selected frequency
    sp = sp.to_numpy()  # Convert index to numpy array

    ints = np.column_stack((sp[:-nD*2+1], ep[nD*2-1:]))   # pairs the start and end indices of the intervals for the given number of days
    
    def apply_realized_estimators_ORS(x):
        start, end = ints[x]
        data = hf[start:end]#.cumsum()
        return RealizedEstimatorsORS(data, nD, NP=False)
    
    #result = np.array([apply_realized_estimators_ORS(x) for x in range(len(ints))])
    result = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4'])
    for x in range(len(ints)):

        result = pd.concat([result if not result.empty else None, apply_realized_estimators_ORS(x)])
    
    return result

# Function to compute realized estimators ORS, need as input log-prices (!) now the updated version uses log returns as inputs
def RealizedEstimatorsORS(x, nD, NP=True):

    N = len(x)   # N = tau * nD, number of all observations; nD - the number of days to be used to get one (!) realized moment.
    tau = N // 2  # Number of HF obs within one LF observation
    #rt = x.diff()                       #np.diff(x)  # Calculate returns
    rt = x                        # in the updated version uses log returns as inputs
    foravg_rt = rt                      # for calculating m1 using the whole series later
    x = x.cumsum()                # get the log prices
    rm = x.rolling(tau).mean()          # pd.Series(x).rolling(tau).mean().values  # Rolling mean, here the rolling mean is only used to compute "y"
    rm = rm[tau-1:-1]       # remove nan in the beginning  # Remove last element to match length, considering at t=N, we only need the moving average of the last time interval.
    rm2 = (x**2).rolling(tau).mean()    # pd.Series(x**2).rolling(tau).mean().values
    rm2 = rm2[tau-1:-1]
    rm3 = (x**3).rolling(tau).mean()    # pd.Series(x**3).rolling(tau).mean().values
    rm3 = rm3[tau-1:-1]
    x = x[tau-1:N-1]     # remove the original elements from 1st to "tau-1"th and the last one because this is used to compute "y", start from InP78 in case tau=78
    rt = rt[tau:]      # Please note the calculation starts from t=79, the first return element is InP79 - InP78, also remove the first nan
    y = x - rm
    z = x**2 - 2*x*rm + rm2
    
    #m1d = rt.sum()
    m1d = foravg_rt.sum() / 2                 # for the average log return of the whole series
    mean = rt.rolling(tau).sum().values[-1]
    
    if NP:
        m2d_NP = np.sum(rt**2) 
        m3d_NP = np.sum(rt**3 + 3 * y * rt**2) 
        m4d_NP = np.sum(rt**4 + 4 * y * rt**3 + 6 * z * rt**2) 
        result = np.column_stack((m1d, m2d_NP, m3d_NP, m4d_NP))
        outcome = pd.DataFrame(result, columns=['m1', 'm2', 'm3', 'm4'], index=[rt.index[-1]])
        return outcome
    else:
        w = x**3 - 3*x**2*rm + 3*x*rm2 - rm3
        m2d = np.sum(rt**2 + 2 * y * rt) 
        m3d = np.sum(rt**3 + 3 * y * rt**2 + 3 * z * rt) 
        m4d = np.sum(rt**4 + 4 * y * rt**3 + 6 * z * rt**2 + 4 * w * rt) 
        result = np.column_stack((m1d, m2d, m3d, m4d))
        outcome = pd.DataFrame(result, columns=['m1', 'm2', 'm3', 'm4'], index=[rt.index[-1]])
        return outcome

# Function to get moments matrix
def rMoments_nc(hfData, method, days_aggregate=22):
    if method == RM_ACJV:
        moments = rMomACJV(hfData, days_aggregate)
    elif method == RM_CL:
        moments = rMomCL(hfData, days_aggregate)
    elif method == RM_NP:
        moments = rMomNP(hfData, days_aggregate)
    elif method == RM_NP_return:
        moments = rMomNP_return(hfData, days_aggregate)
    elif method == RM_ORS:
        moments = rMomORS(hfData, days_aggregate)
    else:
        print("This method for computing moments is not implemented")
        moments = None
    # print(moments)
    return moments

# Function to compute daily returns and realized moments
def rMoments_mvsek(hfData, method, days_aggregate=22, m1zero=True, Sk_out=15, Ku_out=20, ret_nc_mom=False):
    #dret = hfData.resample('D').sum()
    moments = rMoments_nc(hfData, method, days_aggregate)
    
    # rSkKu = moms2skkurt(moments, m1zero) # output centered var, skewness and kurtosis
    rSkKu = cumus2skkurt(moments) # output centered var, skewness and kurtosis
    rr = np.array(rSkKu)                         
    
    # Eliminating outliers in skewness
    ind_big_sk = np.abs(rr[:, 2]) > Sk_out   # set a boolean array
    if ind_big_sk.any():                     # in case there are some outliers
        rr[ind_big_sk, 2] = np.nan
        rr[:, 2] = pd.Series(rr[:, 2]).interpolate().values
        rr = np.column_stack((rr, np.where(rr[:, 2] < 0, -rr[:, 2], 0), np.where(rr[:, 2] > 0, rr[:, 2], 0)))
    
    # Eliminating outliers in kurtosis
    ind_big_ku = rr[:, 3] > Ku_out
    if ind_big_ku.any():
        rr[ind_big_ku, 3] = np.nan
        rr[:, 3] = pd.Series(rr[:, 3]).interpolate().values
    
    if not ret_nc_mom:
        #return rr
        outcome = pd.DataFrame(rr, columns=['rMean', 'rVar', 'rSkew', 'rKurt'], index=moments.index)
        return outcome
    else:
        #return np.column_stack((moments, rr[:, 2], rr[:, 3]))
        outcome = np.column_stack((np.array(moments), rr[:, 2], rr[:, 3]))
        outcome = pd.DataFrame(outcome, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=moments.index)
        return outcome


from scipy.stats import norm

# Cornish-Fisher Expansion using moments up to kurtosis
def CornishFisherExp(alpha, Mu, Sig, Sk, Ku):
    x = norm.ppf(alpha)
    return Mu + Sig * (x + Sk * (x**2 - 1) / 6 + Ku * (x**3 - 3 * x) / 24 + Sk**2 * (5*x - 2 * x**3) / 36)


# Function to compute realized estimators Neuberger & Payne (2020) in the return form
def RealizedEstimatorsNP_return(x, nD):        # use log-prices x as inputs (!) now the updated version uses log returns as inputs
    # x is log price (!) now x is log returns
    N = len(x)   # N = tau * nD, number of all observations; nD - the number of days to be used to get one (!) realized moment.
    tau = N // 2   # Number of HF obs within one LF observation
    #rt = x.diff()                       #np.diff(x)  # Calculate returns
    rt = x                        # in the updated version uses log returns as inputs
    foravg_rt = rt                      # for calculating m1 using the whole series later
    x = x.cumsum()                # get the log prices
    ex = np.exp(x)             # get the prices but still keep the data frame format (not changed to np!)
    x_inv = 1 / ex
    
    rm = x_inv.rolling(tau).mean()          # Rolling mean, here the rolling mean is only used to compute "y1"
    rm = rm[tau-1:-1]       # remove nan in the beginning  # Remove last element to match length, considering at t=N, we only need the moving average of the last time interval.
   
    rm2 = x.rolling(tau).mean()    
    rm2 = rm2[tau-1:-1]

    x = x[tau-1:N-1]
    ex = ex[tau-1:N-1]     # remove the original elements from 1st to "tau-1"th and the last one because this is used to compute "y", start from InP78 in case tau=78
    rt = rt[tau:]      # Please note the calculation starts from t=79, the first return element is InP79 - InP78, also remove the first nan
    y1 = ex * rm - 1
    y2L = 2 * ex * rm - 2 - 2 * x + 2 * rm2
    
    #m1r = (np.exp(rt) - 1).sum()            # for percentage return
    m1r = foravg_rt.sum() / 2                # for the average log return of the whole series
    mean = rt.rolling(tau).sum().values[-1]
    

    m2r_NP = np.sum(2 * (np.exp(rt) - 1 - rt)) 
    m3r_NP = np.sum(6 * ((np.exp(rt) + 1) * rt - 2 * (np.exp(rt) - 1)) + 3 * y1 * 2 * (rt * np.exp(rt) - np.exp(rt) + 1)) 
    m4r_NP = np.sum(12 * (rt**2 + 2 * (np.exp(rt) + 2) * rt - 6 * (np.exp(rt) -  1)) + 
                    4 * y1 * 6 * ((np.exp(rt) + 1) * rt - 2 * (np.exp(rt) - 1)) + 
                    6 * y2L * 2 * (np.exp(rt) - 1 - rt))
    result = np.column_stack((m1r, m2r_NP, m3r_NP, m4r_NP))
    outcome = pd.DataFrame(result, columns=['m1', 'm2', 'm3', 'm4'], index=[rt.index[-1]])

    return outcome
  
def rMomNP_return(hf, nD):
    ep = endpoints(hf, 'D')  # define the endpoints of the data with respect to selected frequency
    ep = ep.to_numpy()  # Convert index to numpy array

    sp = startpoints(hf, 'D')  # define the start points of the data with respect to selected frequency
    sp = sp.to_numpy()  # Convert index to numpy array
    
    ints = np.column_stack((sp[:-nD*2+1], ep[nD*2-1:]))   # pairs the start and end indices of the intervals for the given number of days, i.e., (0,4), (1,5),..., (-4,:)
    
    def apply_realized_estimators_NP_return(x):
        start, end = ints[x]
        data = hf[start:end]#.cumsum()
        return RealizedEstimatorsNP_return(data, nD)
    
    result = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4'])         # define an empty dataframe

    for x in range(len(ints)):

        result = pd.concat([result if not result.empty else None, apply_realized_estimators_NP_return(x)])
    
    return result