import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from datetime import datetime
from SimHestonQE import Heston_QE
from TimeTruncation import align_to_business_days
import seaborn as sns
import matplotlib.pyplot as plt

start_time = datetime.now()
seed_everything(seed=33)

# get a simulated path
print('*****************************************************Simulation*****************************************************************')

time_points = 60 * 22 * 12    
burnin = 10 * 22 * 12
T = 60         
S0 = 100
paths = 1
QE_process = Heston_QE(S0=S0, v0=0.19, kappa=3, theta=0.19, sigma=0.4, mu=0, rho=-0.7, T=T, n=time_points, M=paths)    # the first point is lnS0
QE_process = np.diff(QE_process)                    # to get the difference, i.e., log-returns
QE_process_cut = QE_process[:, burnin:]         # set the first 2 years and 10 months as burnin
QE_process_transform = QE_process_cut.T         # need to transform it to set the index

# Create a DatetimeIndex for every 5 minutes during trading hours
start_date = '2014-07-01'
end_date = '2076-07-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
#time_range = pd.date_range(start='09:30', end='16:05', freq='5T')[:-1]  # Trading hours making it actually end at 16:00

# Create a full DatetimeIndex for all trading days and intraday times
datetime_index = pd.DatetimeIndex([pd.Timestamp(f'{day.date()}') for day in date_range])

# Ensure the datetime_index length matches the data length
time_index = time_points - burnin
datetime_index = datetime_index[:time_index]

# Create the DataFrame
df_logreturn = pd.DataFrame(QE_process_transform, index=datetime_index, columns=[f'Path_{i+1}' for i in range(QE_process_cut.shape[0])])

# make sure the first point is the first business day and the last point is the last business day of a certain month
cut_df_logreturn = align_to_business_days(df_logreturn)
#print(cut_df_logreturn)

# get the monthly return
monthly_logreturn = cut_df_logreturn.resample('M').sum()
data = monthly_logreturn.to_numpy()

# Plotting the distribution with Seaborn (histogram + KDE)

# Disable Seaborn styling if needed
sns.set(style="whitegrid")

sns.histplot(data, label='Empirical', bins=30, kde=True, stat='density', palette=['red'])
#sns.kdeplot(data, color='orange')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Empirical Distribution with KDE')
plt.show()