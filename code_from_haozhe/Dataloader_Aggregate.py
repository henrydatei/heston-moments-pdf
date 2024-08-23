import pickle
import numpy as np
import pandas as pd
import os
from Moments_list import MomentsCIR,getSkFromMoments,getKuFromMoments, MomentsBates

with open('NP_return_22d_kappa3_2yburnout_mu0_v0019_update.pickle', 'rb') as file:
     rm = pickle.load(file)

#print(rm)
print(rm.shape)

# # Check theoretical moments 
print('*****************************************************Theoretical Moments*****************************************************************')
Evp = MomentsBates(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False)   # v0 = 0.1
Evp_nc = MomentsBates(mu = 0, kappa = 3, theta = 0.19, sigma = 0.4, rho=-0.7, lambdaj=0, muj=0, vj=0, t = 1/12, v0 = 0.19, conditional=False, nc=True)   # v0 = 0.1
print(Evp)
print(Evp_nc)
print('\n'*1)

# create an Excel writer object
file_path = os.path.join(os.getcwd(), 'ConvergenceCheck_NP_return_22d_kappa3_2yburnout_mu0_v0019_update.xlsx') 
writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

# get the estimate for 50 simulations
rm_mean_50 = np.mean(rm[:50, :], axis=0).reshape(1,6)  
rm_std_50 = np.std(rm[:50, :], axis=0).reshape(1,6)  
rm_5_50 = np.percentile(rm[:50, :], 5, axis=0).reshape(1,6)
rm_95_50 = np.percentile(rm[:50, :], 95, axis=0).reshape(1,6)   

final_rm_50 = np.concatenate((rm_mean_50, rm_std_50, rm_5_50, rm_95_50), axis=0)
final_rm_50 = pd.DataFrame(final_rm_50, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 50 simulation result*****************************************************************')
print(final_rm_50)
 
final_rm_50.to_excel(writer, sheet_name="50simulations")
print('\n'*1)


# get the estimate for 100 simulations
rm_mean_100 = np.mean(rm[:100, :], axis=0).reshape(1,6)  
rm_std_100 = np.std(rm[:100, :], axis=0).reshape(1,6)  
rm_5_100 = np.percentile(rm[:100, :], 5, axis=0).reshape(1,6)
rm_95_100 = np.percentile(rm[:100, :], 95, axis=0).reshape(1,6)   

final_rm_100 = np.concatenate((rm_mean_100, rm_std_100, rm_5_100, rm_95_100), axis=0)
final_rm_100 = pd.DataFrame(final_rm_100, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 100 simulation result*****************************************************************')
print(final_rm_100)

final_rm_100.to_excel(writer, sheet_name="100simulations")
print('\n'*1)


# get the estimate for 1000 simulations
rm_mean_1000 = np.mean(rm[:1000, :], axis=0).reshape(1,6)  
rm_std_1000 = np.std(rm[:1000, :], axis=0).reshape(1,6)  
rm_5_1000 = np.percentile(rm[:1000, :], 5, axis=0).reshape(1,6)
rm_95_1000 = np.percentile(rm[:1000, :], 95, axis=0).reshape(1,6)   

final_rm_1000 = np.concatenate((rm_mean_1000, rm_std_1000, rm_5_1000, rm_95_1000), axis=0)
final_rm_1000 = pd.DataFrame(final_rm_1000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 1000 simulation result*****************************************************************')
print(final_rm_1000)

final_rm_1000.to_excel(writer, sheet_name="1000simulations")
print('\n'*1)

# get the estimate for 5000 simulations
rm_mean_5000 = np.mean(rm[:5000, :], axis=0).reshape(1,6)  
rm_std_5000 = np.std(rm[:5000, :], axis=0).reshape(1,6)  
rm_5_5000 = np.percentile(rm[:5000, :], 5, axis=0).reshape(1,6)
rm_95_5000 = np.percentile(rm[:5000, :], 95, axis=0).reshape(1,6)   

final_rm_5000 = np.concatenate((rm_mean_5000, rm_std_5000, rm_5_5000, rm_95_5000), axis=0)
final_rm_5000 = pd.DataFrame(final_rm_5000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 5000 simulation result*****************************************************************')
print(final_rm_5000)
 
final_rm_5000.to_excel(writer, sheet_name="5000simulations")
print('\n'*1)


# get the estimate for 10000 simulations
rm_mean_10000 = np.mean(rm[:10000, :], axis=0).reshape(1,6)  
rm_std_10000 = np.std(rm[:10000, :], axis=0).reshape(1,6)  
rm_5_10000 = np.percentile(rm[:10000, :], 5, axis=0).reshape(1,6)
rm_95_10000 = np.percentile(rm[:10000, :], 95, axis=0).reshape(1,6)   

final_rm_10000 = np.concatenate((rm_mean_10000, rm_std_10000, rm_5_10000, rm_95_10000), axis=0)
final_rm_10000 = pd.DataFrame(final_rm_10000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 10000 simulation result*****************************************************************')
print(final_rm_10000)

final_rm_10000.to_excel(writer, sheet_name="10000simulations")
print('\n'*1)


# get the estimate for 50000 simulations
rm_mean_50000 = np.mean(rm[:50000, :], axis=0).reshape(1,6)  
rm_std_50000 = np.std(rm[:50000, :], axis=0).reshape(1,6)  
rm_5_50000 = np.percentile(rm[:50000, :], 5, axis=0).reshape(1,6)
rm_95_50000 = np.percentile(rm[:50000, :], 95, axis=0).reshape(1,6)   

final_rm_50000 = np.concatenate((rm_mean_50000, rm_std_50000, rm_5_50000, rm_95_50000), axis=0)
final_rm_50000 = pd.DataFrame(final_rm_50000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 50000 simulation result*****************************************************************')
print(final_rm_50000)

final_rm_50000.to_excel(writer, sheet_name="50000simulations")
print('\n'*1)


# get the estimate for 100000 simulations
rm_mean_100000 = np.mean(rm[:100000, :], axis=0).reshape(1,6)  
rm_std_100000 = np.std(rm[:100000, :], axis=0).reshape(1,6)  
rm_5_100000 = np.percentile(rm[:100000, :], 5, axis=0).reshape(1,6)
rm_95_100000 = np.percentile(rm[:100000, :], 95, axis=0).reshape(1,6)   

final_rm_100000 = np.concatenate((rm_mean_100000, rm_std_100000, rm_5_100000, rm_95_100000), axis=0)
final_rm_100000 = pd.DataFrame(final_rm_100000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 100000 simulation result*****************************************************************')
print(final_rm_100000)

final_rm_100000.to_excel(writer, sheet_name="100000simulations")
print('\n'*1)


# get the estimate for 200000 simulations
rm_mean_200000 = np.mean(rm[~np.isnan(rm).any(axis=1)], axis=0).reshape(1,6)  
rm_std_200000 = np.std(rm[~np.isnan(rm).any(axis=1)], axis=0).reshape(1,6)  
rm_5_200000 = np.percentile(rm[~np.isnan(rm).any(axis=1)], 5, axis=0).reshape(1,6)
rm_95_200000 = np.percentile(rm[~np.isnan(rm).any(axis=1)], 95, axis=0).reshape(1,6)   

final_rm_200000 = np.concatenate((rm_mean_200000, rm_std_200000, rm_5_200000, rm_95_200000), axis=0)
final_rm_200000 = pd.DataFrame(final_rm_200000, columns=['m1', 'm2', 'm3', 'm4', 'rSkew', 'rKurt'], index=['mean', 'std', '5 percentile', '95 percentile'])    # non-centered case

print('*************************************************Realized Moments 200000 simulation result*****************************************************************')
print(final_rm_200000)
 
final_rm_200000.to_excel(writer, sheet_name="200000simulations")
print('\n'*1)

# Save the Excel file
writer.close()
print('*****************************************************Convergence Check is done*****************************************************************')

na = np.argwhere(np.isnan(rm))
print(na)

rmna = rm[~np.isnan(rm).any(axis=1)]
print(rmna)