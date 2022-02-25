from pymysql import NULL
from global_var import *
import pandas as pd 
import geopandas as gpd
import pickle
import os
import numpy as np
import tqdm

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

## Import and clean 1990 Block, 2010 Tract Covariates
blck_1990 = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_blck_covs/nhgis0033_ds120_1990_block.csv', nrows=20)
print(blck_1990)

## Weight tract covs, construct covs-outcomes for 2015 termination

## Predict KFR for 2035 termination 

