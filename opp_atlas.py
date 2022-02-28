from tabnanny import verbose
from global_var import *
import pandas as pd 
import geopandas as gpd
import pickle
import os
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from joblib import dump, load
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

## Import and merge outcomes
# p_p_p = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_pall.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_pall':'p_p_p'})[['tract','p_p_p']]
# p_p_25 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_p25.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_p25':'p_p_25'})[['tract','p_p_25']]
# p_m_25 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gM_p25.csv').rename(columns={'Household_Income_at_Age_35_rP_gM_p25':'p_m_25'})[['tract','p_m_25']]
# p_f_25 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gF_p25.csv').rename(columns={'Household_Income_at_Age_35_rP_gF_p25':'p_f_25'})[['tract','p_f_25']]

# outcomes = p_p_p.merge(p_p_25, on='tract').merge(p_m_25, on='tract').merge(p_f_25, on='tract')
# outcomes.to_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv', index=False)

# outcomes = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv')
# outcomes = outcomes[outcomes['p_p_p'].notna() & outcomes['p_p_25'].notna() & outcomes['p_m_25'].notna() & outcomes['p_f_25'].notna()] #2010 Tract Codes

## Import 1990-2010 Crosswalk and Overlap weights
# cross = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk/nhgis_blk1990_blk2010_gj.csv')
# cross['TRACT'] = cross['GJOIN2010'].apply(lambda x: str(x[1:3])+str(x[4:7])+str(x[8:14]))
# cross = cross[['GJOIN1990','TRACT','WEIGHT']]
# cross = cross.groupby(['GJOIN1990','TRACT']).sum().reset_index().rename(columns={'GJOIN1990':'GISJOIN'})

# pop = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_block_population/nhgis0032_ds120_1990_block.csv')[['GISJOIN','ET1001']].rename(columns={'ET1001':'POP'})
# merged = cross.merge(pop, how='left', on='GISJOIN')
# merged['WEIGHT'] = merged['WEIGHT']*merged['POP']
# tract = merged.groupby('TRACT').sum().reset_index()[['TRACT','WEIGHT']].rename(columns={'WEIGHT':'TRACT_POP'})
# blocks = merged.merge(tract, how='left', on='TRACT')
# blocks['WEIGHT'] = blocks['WEIGHT']/blocks['TRACT_POP']
# blocks = blocks[['GISJOIN','TRACT','WEIGHT']].rename(columns={'GISJOIN':'1990_BLOCK_GJ','TRACT':'2010_TRACT_GID'})
# blocks.to_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk_weights.csv', index=False)

## Import and Clean 1990 Block
# blck_1990 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_blck_covs/nhgis0033_ds120_1990_block.csv')

# # add density, education
# columns = list(blck_1990.columns)
# aggs = ['EUX','ET2','ET3','ET8','EUG','ESN','ES1','ET0','ESR','ESW','ES4','ETH']
# covs = ['GISJOIN','pop_total','female_perc','white_perc','black_perc','asian_perc','hisp_perc','>65_perc','<18_perc','married_perc','married_families_perc','single_parent_perc',
#         'single_parent_perc_families','singles_perc','own_child_perc','relative_child_perc','foster_child_perc','vacant_houses_perc','owner_houses_perc','white_owner_perc','black_owner_perc',
#         'asian_owner_perc','white_renter_perc','black_renter_perc','asian_renter_perc','<15k_houses_perc','>500k_houses_perc','15-45k_houses_perc','150-500k_houses_perc','house_median','house_agg',
#         'perc_owner_white','perc_owner_black','perc_owner_asian','<100_rent_perc','>1000_rent_perc','100-400_rent_perc','500-1000_rent_perc','rent_median','rent_agg','single_family_home_perc',
#         'single_family_home_attach_perc','2-9_units_perc','10-50_units_perc','mobile_units_perc']

# for agg in aggs:
#     blck_1990[f'{agg}'] = blck_1990[[col for col in columns if f'{agg}' in col]].sum(axis=1)

# blck_1990['pop_total'] = blck_1990['EUX']
# blck_1990['female_perc'] = blck_1990['EUX002']/blck_1990['EUX']
# blck_1990['white_perc'] = blck_1990['ET2001']/blck_1990['ET2']
# blck_1990['black_perc'] = blck_1990['ET2002']/blck_1990['ET2']
# blck_1990['asian_perc'] = blck_1990['ET2004']/blck_1990['ET2']
# blck_1990['hisp_perc'] = (blck_1990['ET2006']+blck_1990['ET2007']+blck_1990['ET2008']+blck_1990['ET2009']+blck_1990['ET2010'])/blck_1990['ET2']
# blck_1990['>65_perc'] = (blck_1990['ET3027']+blck_1990['ET3028']+blck_1990['ET3029']+blck_1990['ET3030']+blck_1990['ET3031'])/blck_1990['ET3']
# blck_1990['<18_perc'] = (blck_1990['ET3001']+blck_1990['ET3002']+blck_1990['ET3003']+blck_1990['ET3004']+blck_1990['ET3005']+blck_1990['ET3006']+blck_1990['ET3007']+blck_1990['ET3008']+blck_1990['ET3009']+blck_1990['ET3010']+blck_1990['ET3011']+blck_1990['ET3012']+blck_1990['ET3013'])/blck_1990['ET3']
# blck_1990['married_perc'] = (blck_1990['ET8003']+blck_1990['ET8004'])/blck_1990['ET8']
# blck_1990['married_families_perc'] = blck_1990['ET8003']/blck_1990['ET8']
# blck_1990['single_parent_perc'] = (blck_1990['ET8005']+blck_1990['ET8007'])/blck_1990['ET8']
# blck_1990['single_parent_perc_families'] = (blck_1990['ET8005']+blck_1990['ET8007'])/(blck_1990['ET8003']+blck_1990['ET8005']+blck_1990['ET8007'])
# blck_1990['singles_perc'] = (blck_1990['ET8001']+blck_1990['ET8002'])/blck_1990['ET8']
# blck_1990['own_child_perc'] = (blck_1990['EUG002']+blck_1990['EUG003']+blck_1990['EUG004']+blck_1990['EUG005']+blck_1990['EUG006']+blck_1990['EUG007']+blck_1990['EUG008'])/blck_1990['EUG']
# blck_1990['relative_child_perc'] = (blck_1990['EUG009']+blck_1990['EUG010']+blck_1990['EUG011']+blck_1990['EUG012']+blck_1990['EUG013']+blck_1990['EUG014']+blck_1990['EUG015'])/blck_1990['EUG']
# blck_1990['foster_child_perc'] = (blck_1990['EUG016']+blck_1990['EUG017']+blck_1990['EUG018']+blck_1990['EUG019']+blck_1990['EUG020']+blck_1990['EUG021']+blck_1990['EUG022'])/blck_1990['EUG']
# blck_1990['vacant_houses_perc'] = blck_1990['ESN002']/blck_1990['ESN']
# blck_1990['owner_houses_perc'] = blck_1990['ES1001']/blck_1990['ES1']
# blck_1990['white_owner_perc'] = blck_1990['ET0001']/blck_1990['ET0']
# blck_1990['black_owner_perc'] = blck_1990['ET0002']/blck_1990['ET0']
# blck_1990['asian_owner_perc'] = blck_1990['ET0004']/blck_1990['ET0']
# blck_1990['white_renter_perc'] = blck_1990['ET0006']/blck_1990['ET0']
# blck_1990['black_renter_perc'] = blck_1990['ET0007']/blck_1990['ET0']
# blck_1990['asian_renter_perc'] = blck_1990['ET0009']/blck_1990['ET0']
# blck_1990['<15k_houses_perc'] = blck_1990['ESR001']/blck_1990['ESR']
# blck_1990['>500k_houses_perc'] = blck_1990['ESR020']/blck_1990['ESR']
# blck_1990['15-45k_houses_perc'] = (blck_1990['ESR002']+blck_1990['ESR003']+blck_1990['ESR004']+blck_1990['ESR005']+blck_1990['ESR006']+blck_1990['ESR007'])/blck_1990['ESR']
# blck_1990['150-500k_houses_perc'] = (blck_1990['ESR014']+blck_1990['ESR015']+blck_1990['ESR016']+blck_1990['ESR017']+blck_1990['ESR018']+blck_1990['ESR019'])/blck_1990['ESR']
# blck_1990['house_median'] = blck_1990['EST001']
# blck_1990['house_agg'] = blck_1990['ESV001']
# blck_1990['perc_owner_white'] = blck_1990['ESW001']/blck_1990['ESW']
# blck_1990['perc_owner_black'] = blck_1990['ESW002']/blck_1990['ESW']
# blck_1990['perc_owner_asian'] = blck_1990['ESW004']/blck_1990['ESW']
# blck_1990['<100_rent_perc'] = blck_1990['ES4001']/blck_1990['ES4']
# blck_1990['>1000_rent_perc'] = blck_1990['ES4016']/blck_1990['ES4']
# blck_1990['100-400_rent_perc'] = (blck_1990['ES4001']+blck_1990['ES4002']+blck_1990['ES4003']+blck_1990['ES4004']+blck_1990['ES4005']+blck_1990['ES4006']+blck_1990['ES4007'])/blck_1990['ES4']
# blck_1990['500-1000_rent_perc'] = (blck_1990['ES4010']+blck_1990['ES4011']+blck_1990['ES4012']+blck_1990['ES4013']+blck_1990['ES4014']+blck_1990['ES4015'])/blck_1990['ES4']
# blck_1990['rent_median'] = blck_1990['ES6001']
# blck_1990['rent_agg'] = blck_1990['ES8001']
# blck_1990['single_family_home_perc'] = blck_1990['ETH001']/blck_1990['ETH']
# blck_1990['single_family_home_attach_perc'] = blck_1990['ETH002']/blck_1990['ETH']
# blck_1990['2-9_units_perc'] = (blck_1990['ETH003']+blck_1990['ETH004']+blck_1990['ETH005'])/blck_1990['ETH']
# blck_1990['10-50_units_perc'] = (blck_1990['ETH006']+blck_1990['ETH007']+blck_1990['ETH008'])/blck_1990['ETH']
# blck_1990['mobile_units_perc'] = blck_1990['ETH009']/blck_1990['ETH']

# blck_1990 = blck_1990[covs]
# cross = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk_weights.csv')
# blck_1990 = cross.merge(blck_1990, how='left', right_on='GISJOIN', left_on='1990_BLOCK_GJ')
# blck_1990 = blck_1990[blck_1990['GISJOIN'].notna() & blck_1990['WEIGHT'].notna()].fillna(0)
# new = blck_1990[['relative_child_perc','relative_child_perc','black_renter_perc','black_renter_perc']].astype(float)
# mult = pd.DataFrame()
# for cov in covs[1:]:
#     new = blck_1990[cov]*blck_1990['WEIGHT']
#     new = new.rename(cov)
#     mult = pd.concat([mult, new], axis=1)
# ids = blck_1990[['2010_TRACT_GID']]
# blck_1990 = pd.concat([ids, mult], axis=1)

# blck_1990 = blck_1990.groupby('2010_TRACT_GID').sum().reset_index()
# blck_1990.to_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs.csv', index=False)

## Learn Tract Outcomes
covs = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs.csv')
outcomes = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv')
outcomes = outcomes.rename(columns = {'tract':'2010_TRACT_GID'})

covs_out = outcomes.merge(covs, how='left', on='2010_TRACT_GID').dropna()
covs = ['pop_total','female_perc','white_perc','black_perc','asian_perc','hisp_perc','>65_perc','<18_perc','married_perc','married_families_perc','single_parent_perc',
        'single_parent_perc_families','singles_perc','own_child_perc','relative_child_perc','foster_child_perc','vacant_houses_perc','owner_houses_perc','white_owner_perc','black_owner_perc',
        'asian_owner_perc','white_renter_perc','black_renter_perc','asian_renter_perc','<15k_houses_perc','>500k_houses_perc','15-45k_houses_perc','150-500k_houses_perc','house_median','house_agg',
        'perc_owner_white','perc_owner_black','perc_owner_asian','<100_rent_perc','>1000_rent_perc','100-400_rent_perc','500-1000_rent_perc','rent_median','rent_agg','single_family_home_perc',
        'single_family_home_attach_perc','2-9_units_perc','10-50_units_perc','mobile_units_perc']

X = covs_out[covs]
y = covs_out[['p_p_p','p_p_25','p_m_25','p_f_25']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=1)

model = Sequential([Dense(y.shape[1], input_dim=X.shape[1])])

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
model.fit(X_train, y_train, epochs=4, validation_data=(X_val, y_val), verbose=1)
model.save(f"full_outcome_model")
print(model.evaluate(X_test, y_test))

## Import and Clean 2010 Tract Covs

## Weight tract covs, construct covs-outcomes for 2015 termination

## Predict KFR for 2035 termination 

