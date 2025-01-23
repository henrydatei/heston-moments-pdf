# Define constants for the realized cumulants
RM_NP_return = 5    # Neuberger and Payne, 2020, in return form

import numpy as np
import pandas as pd

# Function to compute realized moments Neuberger and Payne (2018)
def endpoints(df, freq):
    return df.groupby(df.index.to_period(freq)).apply(lambda x: x.index[-1])                # df.resample(freq).apply(lambda x: x.index[-1])

def startpoints(df, freq):
    return df.groupby(df.index.to_period(freq)).apply(lambda x: x.index[0])

# Function to get cumulants matrix
def rCumulants(hfData: pd.Series, method: int, months_overlap=5):
    if method == RM_NP_return:
        cumulants = rMomNP_return(hfData, months_overlap)
    else:
        print("This method for computing cumulants is not implemented")
        cumulants = None
    return cumulants

# Function to compute realized estimators Neuberger & Payne (2020) in the return form
def RealizedEstimatorsNP_return(x: pd.Series, nM: int):        # use log-prices x as inputs (!) now the updated version uses log returns as inputs
    # x is log price (!) now x is log returns
    N = len(x)   # N = tau * nD, number of all observations; nD - the number of days to be used to get one (!) realized moment.
    tau = N // nM   # Number of HF obs within one LF observation
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
    
    #m1r = (np.exp(rt) - 1).sum() / (nM - 1)  # for percentage return
    m1r = foravg_rt.sum() / nM                # for the average log return of the whole series
    
    m2r_NP = np.sum(2 * (np.exp(rt) - 1 - rt)) / (nM - 1)
    m3r_NP = np.sum(6 * ((np.exp(rt) + 1) * rt - 2 * (np.exp(rt) - 1)) + 3 * y1 * 2 * (rt * np.exp(rt) - np.exp(rt) + 1)) / (nM - 1)
    m4r_NP = np.sum(12 * (rt**2 + 2 * (np.exp(rt) + 2) * rt - 6 * (np.exp(rt) -  1)) + 
                    4 * y1 * 6 * ((np.exp(rt) + 1) * rt - 2 * (np.exp(rt) - 1)) + 
                    6 * y2L * 2 * (np.exp(rt) - 1 - rt)) / (nM - 1)
    result = np.column_stack((m1r, m2r_NP, m3r_NP, m4r_NP))
    outcome = pd.DataFrame(result, columns=['k1', 'k2', 'k3', 'k4'], index=[rt.index[-1]])

    return outcome
  
def rMomNP_return(hf, nM):
    ep = endpoints(hf, 'M')  # define the endpoints of the data with respect to selected frequency
    ep = ep.to_numpy()  # Convert index to numpy array

    sp = startpoints(hf, 'M')  # define the start points of the data with respect to selected frequency
    sp = sp.to_numpy()  # Convert index to numpy array
    
    ints = np.column_stack((sp[:-nM+1], ep[nM-1:]))   # pairs the start and end indices of the intervals for the given number of days, i.e., (0,4), (1,5),..., (-4,:)
    
    def apply_realized_estimators_NP_return(x):
        start, end = ints[x]
        data = hf[start:end]#.cumsum()
        return RealizedEstimatorsNP_return(data, nM)
    
    result = pd.DataFrame(columns=['k1', 'k2', 'k3', 'k4'])         # define an empty dataframe

    for x in range(len(ints)):

        result = pd.concat([result if not result.empty else None, apply_realized_estimators_NP_return(x)])
    
    return result