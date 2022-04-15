from pyparsing import col
from global_var import *
import pandas as pd 
import geopandas as gpd
import pickle
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Import and merge outcomes
p_p_p = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_pall.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_pall':'p_p_p'})[['tract','p_p_p']]
p_p_25 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_p25.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_p25':'p_p_25'})[['tract','p_p_25']]
p_p_50 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_p50.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_p50':'p_p_50'})[['tract','p_p_50']]
p_p_75 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/tract_kfr_rP_gP_p75.csv').rename(columns={'Household_Income_at_Age_35_rP_gP_p75':'p_p_75'})[['tract','p_p_75']]

outcomes = p_p_p.merge(p_p_25, on='tract').merge(p_p_50, on='tract').merge(p_p_75, on='tract')
outcomes = outcomes[outcomes['p_p_p'].notna() & outcomes['p_p_25'].notna() & outcomes['p_p_50'].notna() & outcomes['p_p_75'].notna()] #2010 Tract Codes

outcomes.to_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv', index=False)

# Import 1990-2010 Crosswalk and Overlap weights
cross = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk/nhgis_blk1990_blk2010_gj.csv')
cross['TRACT'] = cross['GJOIN2010'].apply(lambda x: str(x[1:3])+str(x[4:7])+str(x[8:14]))
cross = cross[['GJOIN1990','TRACT','WEIGHT']]
cross = cross.groupby(['GJOIN1990','TRACT']).sum().reset_index().rename(columns={'GJOIN1990':'GISJOIN'})

pop = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_block_population/nhgis0032_ds120_1990_block.csv')[['GISJOIN','ET1001']].rename(columns={'ET1001':'POP'})
merged = cross.merge(pop, how='left', on='GISJOIN')
merged['WEIGHT'] = merged['WEIGHT']*merged['POP']
tract = merged.groupby('TRACT').sum().reset_index()[['TRACT','WEIGHT']].rename(columns={'WEIGHT':'TRACT_POP'})
blocks = merged.merge(tract, how='left', on='TRACT')
blocks['WEIGHT'] = blocks['WEIGHT']/blocks['TRACT_POP']
blocks = blocks[['GISJOIN','TRACT','WEIGHT']].rename(columns={'GISJOIN':'1990_BLOCK_GJ','TRACT':'2010_TRACT_GID'})
blocks.to_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk_weights.csv', index=False)

blckgrps = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk_weights.csv')
blckgrps['1990_BLOCK_GJ'] = blckgrps['1990_BLOCK_GJ'].apply(lambda x: x[:13])
blckgrps = blckgrps.rename(columns={'1990_BLOCK_GJ':'1990_BLCKGRP_GJ'})
blckgrps = blckgrps.groupby(['1990_BLCKGRP_GJ','2010_TRACT_GID']).sum().reset_index()
blckgrps.to_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_blckgrps_crosswalk_weights.csv', index=False)

# Import and Clean 1990 Block
blck_1990 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_blck_covs/nhgis0033_ds120_1990_block.csv', low_memory=False)

columns = list(blck_1990.columns)
aggs = ['EUX','ET2','ET3','ET8','EUG','ESN','ES1','ET0','ESR','ESW','ES4','ETH']
covs = ['GISJOIN','pop_total','female_perc','white_perc','black_perc','asian_perc','hisp_perc','>65_perc','<18_perc','married_perc','married_families_perc','single_parent_perc',
        'single_parent_perc_families','singles_perc','own_child_perc','relative_child_perc','foster_child_perc','vacant_houses_perc','owner_houses_perc','white_owner_perc','black_owner_perc',
        'asian_owner_perc','white_renter_perc','black_renter_perc','asian_renter_perc','<15k_houses_perc','>500k_houses_perc','15-45k_houses_perc','150-500k_houses_perc','house_median','house_agg',
        'perc_owner_white','perc_owner_black','perc_owner_asian','<100_rent_perc','>1000_rent_perc','100-400_rent_perc','500-1000_rent_perc','rent_median','rent_agg','single_family_home_perc',
        'single_family_home_attach_perc','2-9_units_perc','10-50_units_perc','mobile_units_perc']

for agg in aggs:
    blck_1990[f'{agg}'] = blck_1990[[col for col in columns if f'{agg}' in col]].sum(axis=1)

blck_1990['pop_total'] = blck_1990['EUX']
blck_1990['female_perc'] = blck_1990['EUX002']/blck_1990['EUX']
blck_1990['white_perc'] = blck_1990['ET2001']/blck_1990['ET2']
blck_1990['black_perc'] = blck_1990['ET2002']/blck_1990['ET2']
blck_1990['asian_perc'] = blck_1990['ET2004']/blck_1990['ET2']
blck_1990['hisp_perc'] = (blck_1990['ET2006']+blck_1990['ET2007']+blck_1990['ET2008']+blck_1990['ET2009']+blck_1990['ET2010'])/blck_1990['ET2']
blck_1990['>65_perc'] = (blck_1990['ET3027']+blck_1990['ET3028']+blck_1990['ET3029']+blck_1990['ET3030']+blck_1990['ET3031'])/blck_1990['ET3']
blck_1990['<18_perc'] = (blck_1990['ET3001']+blck_1990['ET3002']+blck_1990['ET3003']+blck_1990['ET3004']+blck_1990['ET3005']+blck_1990['ET3006']+blck_1990['ET3007']+blck_1990['ET3008']+blck_1990['ET3009']+blck_1990['ET3010']+blck_1990['ET3011']+blck_1990['ET3012']+blck_1990['ET3013'])/blck_1990['ET3']
blck_1990['married_perc'] = (blck_1990['ET8003']+blck_1990['ET8004'])/blck_1990['ET8']
blck_1990['married_families_perc'] = blck_1990['ET8003']/blck_1990['ET8']
blck_1990['single_parent_perc'] = (blck_1990['ET8005']+blck_1990['ET8007'])/blck_1990['ET8']
blck_1990['single_parent_perc_families'] = (blck_1990['ET8005']+blck_1990['ET8007'])/(blck_1990['ET8003']+blck_1990['ET8005']+blck_1990['ET8007'])
blck_1990['singles_perc'] = (blck_1990['ET8001']+blck_1990['ET8002'])/blck_1990['ET8']
blck_1990['own_child_perc'] = (blck_1990['EUG002']+blck_1990['EUG003']+blck_1990['EUG004']+blck_1990['EUG005']+blck_1990['EUG006']+blck_1990['EUG007']+blck_1990['EUG008'])/blck_1990['EUG']
blck_1990['relative_child_perc'] = (blck_1990['EUG009']+blck_1990['EUG010']+blck_1990['EUG011']+blck_1990['EUG012']+blck_1990['EUG013']+blck_1990['EUG014']+blck_1990['EUG015'])/blck_1990['EUG']
blck_1990['foster_child_perc'] = (blck_1990['EUG016']+blck_1990['EUG017']+blck_1990['EUG018']+blck_1990['EUG019']+blck_1990['EUG020']+blck_1990['EUG021']+blck_1990['EUG022'])/blck_1990['EUG']
blck_1990['vacant_houses_perc'] = blck_1990['ESN002']/blck_1990['ESN']
blck_1990['owner_houses_perc'] = blck_1990['ES1001']/blck_1990['ES1']
blck_1990['white_owner_perc'] = blck_1990['ET0001']/blck_1990['ET0']
blck_1990['black_owner_perc'] = blck_1990['ET0002']/blck_1990['ET0']
blck_1990['asian_owner_perc'] = blck_1990['ET0004']/blck_1990['ET0']
blck_1990['white_renter_perc'] = blck_1990['ET0006']/blck_1990['ET0']
blck_1990['black_renter_perc'] = blck_1990['ET0007']/blck_1990['ET0']
blck_1990['asian_renter_perc'] = blck_1990['ET0009']/blck_1990['ET0']
blck_1990['<15k_houses_perc'] = blck_1990['ESR001']/blck_1990['ESR']
blck_1990['>500k_houses_perc'] = blck_1990['ESR020']/blck_1990['ESR']
blck_1990['15-45k_houses_perc'] = (blck_1990['ESR002']+blck_1990['ESR003']+blck_1990['ESR004']+blck_1990['ESR005']+blck_1990['ESR006']+blck_1990['ESR007'])/blck_1990['ESR']
blck_1990['150-500k_houses_perc'] = (blck_1990['ESR014']+blck_1990['ESR015']+blck_1990['ESR016']+blck_1990['ESR017']+blck_1990['ESR018']+blck_1990['ESR019'])/blck_1990['ESR']
blck_1990['house_median'] = blck_1990['EST001']
blck_1990['house_agg'] = blck_1990['ESV001']
blck_1990['perc_owner_white'] = blck_1990['ESW001']/blck_1990['ESW']
blck_1990['perc_owner_black'] = blck_1990['ESW002']/blck_1990['ESW']
blck_1990['perc_owner_asian'] = blck_1990['ESW004']/blck_1990['ESW']
blck_1990['<100_rent_perc'] = blck_1990['ES4001']/blck_1990['ES4']
blck_1990['>1000_rent_perc'] = blck_1990['ES4016']/blck_1990['ES4']
blck_1990['100-400_rent_perc'] = (blck_1990['ES4001']+blck_1990['ES4002']+blck_1990['ES4003']+blck_1990['ES4004']+blck_1990['ES4005']+blck_1990['ES4006']+blck_1990['ES4007'])/blck_1990['ES4']
blck_1990['500-1000_rent_perc'] = (blck_1990['ES4010']+blck_1990['ES4011']+blck_1990['ES4012']+blck_1990['ES4013']+blck_1990['ES4014']+blck_1990['ES4015'])/blck_1990['ES4']
blck_1990['rent_median'] = blck_1990['ES6001']
blck_1990['rent_agg'] = blck_1990['ES8001']
blck_1990['single_family_home_perc'] = blck_1990['ETH001']/blck_1990['ETH']
blck_1990['single_family_home_attach_perc'] = blck_1990['ETH002']/blck_1990['ETH']
blck_1990['2-9_units_perc'] = (blck_1990['ETH003']+blck_1990['ETH004']+blck_1990['ETH005'])/blck_1990['ETH']
blck_1990['10-50_units_perc'] = (blck_1990['ETH006']+blck_1990['ETH007']+blck_1990['ETH008'])/blck_1990['ETH']
blck_1990['mobile_units_perc'] = blck_1990['ETH009']/blck_1990['ETH']

blck_1990 = blck_1990[covs]
cross = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_block_crosswalk_weights.csv')
blck_1990 = cross.merge(blck_1990, how='left', right_on='GISJOIN', left_on='1990_BLOCK_GJ')
blck_1990 = blck_1990[blck_1990['GISJOIN'].notna() & blck_1990['WEIGHT'].notna()]

tracts = set(blck_1990['2010_TRACT_GID'])
final = pd.DataFrame()
for tract in tqdm(tracts):
    blck_1990['bool'] = blck_1990['2010_TRACT_GID']==tract
    mult = pd.DataFrame()
    for cov in covs[1:]:
        new = blck_1990.loc[blck_1990['bool'],[cov,'WEIGHT']]
        new.loc[:,'WEIGHT'] = new['WEIGHT']/(new[new[cov].notna()]['WEIGHT'].sum())
        new.loc[:, cov] = new[cov]*new['WEIGHT']
        mult = pd.concat([mult, new[cov]], axis=1)
    ids = blck_1990.loc[blck_1990['bool'],:][['2010_TRACT_GID']]
    tr = pd.concat([ids, mult], axis=1)
    final = pd.concat([final,tr])

blck_1990 = final.groupby('2010_TRACT_GID').sum().reset_index()
blck_1990.to_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs.csv', index=False)

# Import and Clean 1990 Blckgrp
blckgrp_1990 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_blckgrp_covs/nhgis0056_ds123_1990_blck_grp.csv', low_memory=False)

covs = ['GISJOIN','>1hr_comm_perc','<20_comm_perc','private_school_perc','grad_perc','college_perc','some_college_perc','hs_drop_perc','<15k_perc','35-75k_perc','>125k_perc','public_assist_perc','perc_fam_poor','foreign_born_perc']

aggs = ['E3W0','E33','E4T','E3N']
for agg in aggs:
    blckgrp_1990[f'{agg}'] = blckgrp_1990[[col for col in blckgrp_1990.columns if f'{agg}' in col]].sum(axis=1)

blckgrp_1990['>1hr_comm_perc'] = (blckgrp_1990['E3W011']+blckgrp_1990['E3W012'])/blckgrp_1990['E3W0']
blckgrp_1990['<20_comm_perc'] = (blckgrp_1990['E3W004']+blckgrp_1990['E3W003']+blckgrp_1990['E3W002']+blckgrp_1990['E3W001'])/blckgrp_1990['E3W0']
blckgrp_1990['private_school_perc'] = blckgrp_1990['E30004']/(blckgrp_1990['E30004']+blckgrp_1990['E30003'])
blckgrp_1990['grad_perc'] = blckgrp_1990['E33007']/blckgrp_1990['E33']
blckgrp_1990['college_perc'] = blckgrp_1990['E33006']/blckgrp_1990['E33']
blckgrp_1990['some_college_perc'] = (blckgrp_1990['E33004']+blckgrp_1990['E33005'])/blckgrp_1990['E33']
blckgrp_1990['hs_drop_perc'] = (blckgrp_1990['E33001']+blckgrp_1990['E33002'])/blckgrp_1990['E33']
blckgrp_1990['<15k_perc'] = (blckgrp_1990['E4T001']+blckgrp_1990['E4T002']+blckgrp_1990['E4T003']+blckgrp_1990['E4T004'])/blckgrp_1990['E4T']
blckgrp_1990['35-75k_perc'] = (blckgrp_1990['E4T013']+blckgrp_1990['E4T014']+blckgrp_1990['E4T015']+blckgrp_1990['E4T016']+blckgrp_1990['E4T017']+blckgrp_1990['E4T018']+blckgrp_1990['E4T019']+blckgrp_1990['E4T020']+blckgrp_1990['E4T021'])/blckgrp_1990['E4T']
blckgrp_1990['>125k_perc'] = (blckgrp_1990['E4T024']+blckgrp_1990['E4T025'])/blckgrp_1990['E4T']
blckgrp_1990['public_assist_perc'] = blckgrp_1990['E5A001']/(blckgrp_1990['E5A001']+blckgrp_1990['E5A002'])
blckgrp_1990['perc_fam_poor'] = (blckgrp_1990['E1E013']+blckgrp_1990['E1E014']+blckgrp_1990['E1E015']+blckgrp_1990['E1E017']+blckgrp_1990['E1E018']+blckgrp_1990['E1E019']+blckgrp_1990['E1E021']+blckgrp_1990['E1E022']+blckgrp_1990['E1E022'])/(blckgrp_1990['E1E001']+blckgrp_1990['E1E002']+blckgrp_1990['E1E003']+blckgrp_1990['E1E005']+blckgrp_1990['E1E006']+blckgrp_1990['E1E007']+blckgrp_1990['E1E009']+blckgrp_1990['E1E010']+blckgrp_1990['E1E011']+blckgrp_1990['E1E013']+blckgrp_1990['E1E014']+blckgrp_1990['E1E015']+blckgrp_1990['E1E017']+blckgrp_1990['E1E018']+blckgrp_1990['E1E019']+blckgrp_1990['E1E021']+blckgrp_1990['E1E022']+blckgrp_1990['E1E022'])
blckgrp_1990['foreign_born_perc'] = blckgrp_1990['E3N009']/blckgrp_1990['E3N']

blckgrp_1990 = blckgrp_1990[covs]
cross = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_2010_blckgrps_crosswalk_weights.csv')
blckgrp_1990 = cross.merge(blckgrp_1990, how='left', right_on='GISJOIN', left_on='1990_BLCKGRP_GJ')
blckgrp_1990 = blckgrp_1990[blckgrp_1990['GISJOIN'].notna() & blckgrp_1990['WEIGHT'].notna()]

tracts = set(blckgrp_1990['2010_TRACT_GID'])
final = pd.DataFrame()
for tract in tqdm(tracts):
    blckgrp_1990['bool'] = blckgrp_1990['2010_TRACT_GID']==tract
    mult = pd.DataFrame()
    for cov in covs[1:]:
        new = blckgrp_1990.loc[blckgrp_1990['bool'],[cov,'WEIGHT']]
        new.loc[:,'WEIGHT'] = new['WEIGHT']/(new[new[cov].notna()]['WEIGHT'].sum())
        new.loc[:, cov] = new[cov]*new['WEIGHT']
        mult = pd.concat([mult, new[cov]], axis=1)
    ids = blckgrp_1990.loc[blckgrp_1990['bool'],:][['2010_TRACT_GID']]
    tr = pd.concat([ids, mult], axis=1)
    final = pd.concat([final,tr])

blckgrp_1990 = final.groupby('2010_TRACT_GID').sum().reset_index()
blckgrp_1990.to_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_2_covs.csv', index=False)

blck_1990_data = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs.csv')
blckgrp_1990_data = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_2_covs.csv')

tracts = pd.merge(blck_1990_data, blckgrp_1990_data, how='inner', on='2010_TRACT_GID')

tracts.to_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs_final.csv', index=False)

# Import and Clean 2010 Block Covs
covs = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs/nhgis0044_ds176_20105_blck_grp.csv', encoding='latin-1')
covs2 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs/nhgis0044_ds172_2010_blck_grp.csv', encoding='latin-1')
covs3 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs/nhgis0053_ds176_20105_blck_grp.csv', encoding='latin-1')

covs4 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs/nhgis0057_ds177_20105_tract.csv', encoding='latin-1', nrows=20000).rename(columns={'GISJOIN':'GISJOIN_TRACT'})
covs4['foreign_born_perc'] = (covs4['JV8E005']+covs4['JV8E006'])/(covs4['JV8E002']+covs4['JV8E003']+covs4['JV8E004']+covs4['JV8E005']+covs4['JV8E006'])
covs4 = covs4[['GISJOIN_TRACT','foreign_born_perc']]

tract_2010 = covs.merge(covs2, how ='inner', on='GISJOIN').merge(covs3, how ='inner', on='GISJOIN')
tract_2010['GISJOIN_TRACT'] = tract_2010['GISJOIN'].apply(lambda x: x[:14])

covs = ['GISJOIN','pop_total','female_perc','white_perc','black_perc','asian_perc','hisp_perc','>65_perc','<18_perc','married_perc','married_families_perc','single_parent_perc',
        'single_parent_perc_families','singles_perc','own_child_perc','relative_child_perc','foster_child_perc','vacant_houses_perc','owner_houses_perc','white_owner_perc','black_owner_perc',
        'asian_owner_perc','white_renter_perc','black_renter_perc','asian_renter_perc','<15k_houses_perc','>500k_houses_perc','15-45k_houses_perc','150-500k_houses_perc','house_median','house_agg',
        'perc_owner_white','perc_owner_black','perc_owner_asian','<100_rent_perc','>1000_rent_perc','100-400_rent_perc','500-1000_rent_perc','rent_median','rent_agg','single_family_home_perc',
        'single_family_home_attach_perc','2-9_units_perc','10-50_units_perc','mobile_units_perc','>1hr_comm_perc','<20_comm_perc','private_school_perc','grad_perc','college_perc','some_college_perc','hs_drop_perc',
        '<15k_perc','35-75k_perc','>125k_perc','public_assist_perc','perc_fam_poor','foreign_born_perc']

tract_2010['pop_total'] = tract_2010['H7Z001']
tract_2010['female_perc'] = tract_2010['H76026']/tract_2010['H76001']
tract_2010['white_perc'] = tract_2010['H7Z003']/tract_2010['H7Z001']
tract_2010['black_perc'] = tract_2010['H7Z004']/tract_2010['H7Z001']
tract_2010['asian_perc'] = tract_2010['H7Z006']/tract_2010['H7Z001']
tract_2010['hisp_perc'] = tract_2010['H7Z010']/tract_2010['H7Z001']
tract_2010['>65_perc'] = (tract_2010['H76020']+tract_2010['H76021']+tract_2010['H76022']+tract_2010['H76023']+tract_2010['H76024']+tract_2010['H76025']+tract_2010['H76044']+tract_2010['H76045']+tract_2010['H76046']+tract_2010['H76047']+tract_2010['H76048']+tract_2010['H76049'])/tract_2010['H76001']
tract_2010['<18_perc'] = (tract_2010['H76003']+tract_2010['H76004']+tract_2010['H76005']+tract_2010['H76006']+tract_2010['H76027']+tract_2010['H76028']+tract_2010['H76029']+tract_2010['H76030'])/tract_2010['H76001']
tract_2010['married_perc'] = tract_2010['H8D007']/tract_2010['H8D001']
tract_2010['married_families_perc'] = tract_2010['H8D008']/tract_2010['H8D001']
tract_2010['single_parent_perc'] = (tract_2010['H8D012']+tract_2010['H8D015'])/tract_2010['H8D001']
tract_2010['single_parent_perc_families'] = (tract_2010['H8D012']+tract_2010['H8D015'])/(tract_2010['H8D012']+tract_2010['H8D015']+tract_2010['H8D008'])
tract_2010['singles_perc'] = (tract_2010['H8D003']+tract_2010['H8D004'])/tract_2010['H8D001']
tract_2010['own_child_perc'] = tract_2010['H8Q005']/tract_2010['H8Q001']
tract_2010['relative_child_perc'] = tract_2010['H8Q013']/tract_2010['H8Q001']
tract_2010['foster_child_perc'] = tract_2010['H8Q021']/tract_2010['H8Q001']
tract_2010['vacant_houses_perc'] = tract_2010['IFE003']/tract_2010['IFE001']
tract_2010['owner_houses_perc'] = (tract_2010['IFF002']+tract_2010['IFF003'])/tract_2010['IFF001']
tract_2010['<15k_houses_perc'] = (tract_2010['JTGE002']+tract_2010['JTGE003']+tract_2010['JTGE004']+tract_2010['JTGE005'])/tract_2010['JTGE001']
tract_2010['>500k_houses_perc'] = (tract_2010['JTGE024']+tract_2010['JTGE025'])/tract_2010['JTGE001']
tract_2010['15-45k_houses_perc'] = (tract_2010['JTGE006']+tract_2010['JTGE007']+tract_2010['JTGE008']+tract_2010['JTGE009']+tract_2010['JTGE010']+tract_2010['JTGE011']+tract_2010['JTGE012'])/tract_2010['JTGE001']
tract_2010['150-500k_houses_perc'] = (tract_2010['JTGE020']+tract_2010['JTGE021']+tract_2010['JTGE022']+tract_2010['JTGE023']+tract_2010['JTGE024'])/tract_2010['JTGE001']
tract_2010['house_median'] = tract_2010['JTIE001']/1.67
tract_2010['house_agg'] = tract_2010['JTKE001']/1.67
tract_2010['<100_rent_perc'] = (tract_2010['JSXM003']+tract_2010['JSXM004']+tract_2010['JSXM005'])/tract_2010['JSXM002']
tract_2010['>1000_rent_perc'] = tract_2010['JSXM023']/tract_2010['JSXM002']
tract_2010['100-400_rent_perc'] = (tract_2010['JSXM006']+tract_2010['JSXM007']+tract_2010['JSXM008']+tract_2010['JSXM009']+tract_2010['JSXM010']+tract_2010['JSXM011']+tract_2010['JSXM012']+tract_2010['JSXM013']+tract_2010['JSXM014']+tract_2010['JSXM015'])/tract_2010['JSXM002']
tract_2010['500-1000_rent_perc'] = (tract_2010['JSXM019']+tract_2010['JSXM020']+tract_2010['JSXM021']+tract_2010['JSXM022'])/tract_2010['JSXM002']
tract_2010['rent_median'] = tract_2010['JSZM001']/1.67
tract_2010['rent_agg'] = tract_2010['JS1M001']/1.67
tract_2010['single_family_home_perc'] = tract_2010['JSAM002']/tract_2010['JSAM001']
tract_2010['single_family_home_attach_perc'] = tract_2010['JSAM003']/tract_2010['JSAM001']
tract_2010['2-9_units_perc'] = (tract_2010['JSAM004']+tract_2010['JSAM005']+tract_2010['JSAM006'])/tract_2010['JSAM001']
tract_2010['10-50_units_perc'] = (tract_2010['JSAM007']+tract_2010['JSAM008']+tract_2010['JSAM009'])/tract_2010['JSAM001']
tract_2010['mobile_units_perc'] = (tract_2010['JSAM010']+tract_2010['JSAM011'])/tract_2010['JSAM001']
tract_2010['white_owner_perc'] = tract_2010['IFP003']/tract_2010['IFP001']
tract_2010['black_owner_perc'] = tract_2010['IFP004']/tract_2010['IFP001']
tract_2010['asian_owner_perc'] = tract_2010['IFP006']/tract_2010['IFP001']
tract_2010['white_renter_perc'] = tract_2010['IFP011']/tract_2010['IFP001']
tract_2010['black_renter_perc'] = tract_2010['IFP012']/tract_2010['IFP001']
tract_2010['asian_renter_perc'] = tract_2010['IFP014']/tract_2010['IFP001']
tract_2010['perc_owner_white'] = tract_2010['IFP003']/tract_2010['IFP002']
tract_2010['perc_owner_black'] = tract_2010['IFP004']/tract_2010['IFP002']
tract_2010['perc_owner_asian'] = tract_2010['IFP006']/tract_2010['IFP002']
tract_2010['>1hr_comm_perc'] = (tract_2010['JM2E012']+tract_2010['JM2E013'])/tract_2010['JM2E001']
tract_2010['<20_comm_perc'] = (tract_2010['JM2E005']+tract_2010['JM2E003']+tract_2010['JM2E002'])/tract_2010['JM2E001']
tract_2010['private_school_perc'] = (tract_2010['JNYE012']+tract_2010['JNYE015']+tract_2010['JNYE018']+tract_2010['JNYE036']+tract_2010['JNYE039']+tract_2010['JNYE041'])/(tract_2010['JNYE010']+tract_2010['JNYE013']+tract_2010['JNYE016']+tract_2010['JNYE034']+tract_2010['JNYE037']+tract_2010['JNYE040'])
tract_2010['grad_perc'] = (tract_2010['JN9E016']+tract_2010['JN9E017']+tract_2010['JN9E018']+tract_2010['JN9E033']+tract_2010['JN9E034']+tract_2010['JN9E035'])/tract_2010['JN9E001']
tract_2010['college_perc'] = (tract_2010['JN9E015']+tract_2010['JN9E032'])/tract_2010['JN9E001']
tract_2010['some_college_perc'] = (tract_2010['JN9E012']+tract_2010['JN9E013']+tract_2010['JN9E014']+tract_2010['JN9E029']+tract_2010['JN9E030']+tract_2010['JN9E031'])/tract_2010['JN9E001']
tract_2010['hs_drop_perc'] = (tract_2010['JN9E020']+tract_2010['JN9E021']+tract_2010['JN9E022']+tract_2010['JN9E023']+tract_2010['JN9E024']+tract_2010['JN9E025']+tract_2010['JN9E026']+tract_2010['JN9E027']+tract_2010['JN9E003']+tract_2010['JN9E004']+tract_2010['JN9E005']+tract_2010['JN9E006']+tract_2010['JN9E007']+tract_2010['JN9E008']+tract_2010['JN9E009']+tract_2010['JN9E010'])/tract_2010['JN9E001']
tract_2010['<15k_perc'] = (tract_2010['JOHE002']+tract_2010['JOHE003']+tract_2010['JOHE004']+tract_2010['JOHE005'])/tract_2010['JOHE001']
tract_2010['35-75k_perc'] = (tract_2010['JOHE012']+tract_2010['JOHE013']+tract_2010['JOHE014'])/tract_2010['JOHE001']
tract_2010['>125k_perc'] = (tract_2010['JOHE017'])/tract_2010['JOHE001']
tract_2010['public_assist_perc'] = tract_2010['JPBE002']/tract_2010['JPBE001']
tract_2010['perc_fam_poor'] = tract_2010['JODE002']/tract_2010['JODE001']

tract_2010  = tract_2010.merge(covs4, how='left', on='GISJOIN_TRACT')
tract_2010[covs].to_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs.csv', index=False)

tract_2010['GISJOIN_TRACT'] = tract_2010.apply(lambda x: x['GISJOIN'][:-1], axis=1)
tracts = tract_2010.groupby('GISJOIN_TRACT').sum().reset_index()[['GISJOIN_TRACT','pop_total']].rename(columns={'pop_total':'tract_total'})
tract_2010 = tract_2010.merge(tracts, how='left', on='GISJOIN_TRACT')
tract_2010['WEIGHT'] = tract_2010['pop_total']/tract_2010['tract_total']

tracts = set(tract_2010['GISJOIN_TRACT'])
final = pd.DataFrame()
for tract in tqdm(tracts):
    tract_2010['bool'] = tract_2010['GISJOIN_TRACT']==tract
    mult = pd.DataFrame()
    for cov in covs[1:-1]:
        new = tract_2010.loc[tract_2010['bool'],[cov,'WEIGHT']]
        new.loc[:,'WEIGHT'] = new['WEIGHT']/(new[new[cov].notna()]['WEIGHT'].sum())
        new.loc[:, cov] = new[cov]*new['WEIGHT']
        mult = pd.concat([mult, new[cov]], axis=1)
    ids = tract_2010.loc[tract_2010['bool'],:][['GISJOIN_TRACT']]
    tr = pd.concat([ids, mult], axis=1)
    final = pd.concat([final,tr])

tract_2010 = final.groupby('GISJOIN_TRACT').sum().reset_index()
tract_2010  = tract_2010.merge(covs4, how='left', on='GISJOIN_TRACT')
tract_2010.to_csv(f'{S3_PATH}opp_atlas/csv/2010_tract_covs.csv', index=False)

# Learn Tract Outcomes 
covariates = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs_final.csv')

outcomes = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv')
outcomes = outcomes.rename(columns = {'tract':'2010_TRACT_GID'})

covs_out = outcomes.merge(covariates, how='left', on='2010_TRACT_GID').dropna()

outcomes = ['p_p_p','p_p_25','p_p_50','p_p_75',]
covs = [feature for feature in list(covs_out.columns) if feature not in outcomes and feature!='2010_TRACT_GID']

y = covs_out[outcomes]
X = covs_out[covs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

X_poly = PolynomialFeatures(2, interaction_only=True).fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.10)

X_z = StandardScaler().fit_transform(X_poly)
X_train_z, X_test_z, y_train_z, y_test_z = train_test_split(X_z, y, test_size=0.10)

for outcome in outcomes:
    print(outcome)

    linear = LinearRegression().fit(X_train, y_train[outcome].ravel())
    print('Linear')
    print(explained_variance_score( y_test[outcome].ravel(), linear.predict(X_test) ), mean_squared_error( y_test[outcome].ravel(), linear.predict(X_test) ))

    linear_poly = LinearRegression().fit(X_train_poly, y_train_poly[outcome].ravel())
    print('Linear Poly')
    print(explained_variance_score( y_test_poly[outcome].ravel(), linear_poly.predict(X_test_poly) ), mean_squared_error( y_test_poly[outcome].ravel(), linear_poly.predict(X_test_poly) ))

    elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=4).fit(X_train_poly, y_train_poly[outcome].ravel())
    print('Elastic')
    print(explained_variance_score( y_test_poly[outcome].ravel(), elastic.predict(X_test_poly) ), mean_squared_error( y_test_poly[outcome].ravel(), elastic.predict(X_test_poly) ))

    neural = MLPRegressor(hidden_layer_sizes=(10,10)).fit(X_train_poly, y_train_poly[outcome].ravel())
    print('Neural')
    print(explained_variance_score( y_test_poly[outcome].ravel(), neural.predict(X_test_poly) ),mean_squared_error( y_test_poly[outcome].ravel(), neural.predict(X_test_poly) ))

    tree = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1).fit(X_train_z, y_train_z[outcome].ravel())
    print('Nearest Neighbors')
    print(explained_variance_score( y_test_z[outcome].ravel(), tree.predict(X_test_z) ),mean_squared_error( y_test_z[outcome].ravel(), tree.predict(X_test_z) ))
    dump(tree, f'{outcome}_tree.joblib')
    S3FS.put(f'{outcome}_tree.joblib', f'{S3_PATH}opp_atlas/models/{outcome}_tree.joblib')

# Create Tract-PLaces (i.e. add UA and Place Codes)
S3FS.get(f'{S3_PATH}gis/gis/2010_gini_base.gpkg')
gini = gpd.read_file('2010_gini_base.gpkg')[['GISJOIN_PLACE','GISJOIN_BLCKGRP','UACODE','source']]
gini['GISJOIN_TRACT'] = gini['GISJOIN_BLCKGRP'].apply(lambda x: x[:14])
gini = gini[['GISJOIN_PLACE','GISJOIN_TRACT','UACODE','source']].drop_duplicates().reset_index(drop=True)

tract_2010 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_tract_covs.csv')
tract_2010 = gini.merge(tract_2010, how='left', on='GISJOIN_TRACT')
tract_2010.to_csv(f'{S3_PATH}opp_atlas/csv/2010_tract_covs.csv', index=False)

# Create Surrogate Indices
og = list(pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs_final.csv', nrows=1).columns)[1:]

tracts = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_tract_covs.csv')
tracts = tracts.dropna()

X = tracts[og]
X_poly = PolynomialFeatures(2, interaction_only=True).fit_transform(X)
X_z = StandardScaler().fit_transform(X_poly)

outcomes = ['p_p_p','p_p_25','p_p_50','p_p_75']
for outcome in outcomes:
    # S3FS.get(f'{S3_PATH}opp_atlas/models/{outcome}_tree.joblib', f'{outcome}_tree.joblib')
    linear = load(f'{outcome}_tree.joblib')
    tracts[outcome] = np.nan
    tracts.loc[:,outcome] = linear.predict(X_z)

tracts.to_csv(f'{S3_PATH}opp_atlas/csv/surrogates_2010.csv', index=False)