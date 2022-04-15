from enum import auto
from pydoc import describe
from pyparsing import col
from global_var import *
import geopandas as gpd
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm
from tqdm import tqdm
import stata_setup
stata_setup.config('/Applications/Stata/', 'be')
from pystata import stata

## Get Place Covariates
# S3FS.get(f'{S3_PATH}gis/gis/2010_gini_base.gpkg', '2010_gini_base.gpkg')
juris = gpd.read_file('2010_gini_base.gpkg').to_crs("EPSG:3347")
blckgrps = juris[['GISJOIN_BLCKGRP','area']].groupby('GISJOIN_BLCKGRP').sum().reset_index().rename(columns={'area':'blckgrp_area'})
juris = juris.merge(blckgrps, how='left', on='GISJOIN_BLCKGRP')
juris['weight'] = juris['area']/juris['blckgrp_area']
juris = juris[['GISJOIN_PLACE','GISJOIN_BLCKGRP','UACODE','source','weight']]

blocks = pd.read_csv(f'{S3_PATH}opp_atlas/csv/2010_blckgrp_covs.csv').rename(columns={'GISJOIN':'GISJOIN_BLCKGRP'})
juris = juris.merge(blocks, how='left', on='GISJOIN_BLCKGRP')

columns = list(juris.columns)
remove_column = ['GISJOIN_PLACE','GISJOIN_BLCKGRP','UACODE','source','weight']
for col in remove_column:
    columns.remove(col)
for column in columns:
    juris[column] = juris[column]*juris['weight']

places = juris[['GISJOIN_PLACE','pop_total']].groupby('GISJOIN_PLACE').sum().reset_index().rename(columns={'pop_total':'place_pop'})
juris = juris.merge(places, how='left', on='GISJOIN_PLACE')
juris['weight'] = juris['pop_total']/juris['place_pop']
juris = juris.drop(columns=['place_pop'])

columns = list(juris.columns)
remove_column = ['GISJOIN_PLACE','GISJOIN_BLCKGRP','UACODE','source','weight','pop_total','house_agg','rent_agg']
for col in remove_column:
    columns.remove(col)
for column in columns:
    juris[column] = juris[column]*juris['weight']

juris = juris.groupby(['GISJOIN_PLACE','UACODE','source']).sum().reset_index()
juris = juris.drop(columns=['weight'])
juris = juris[juris['pop_total']!=0]

juris.to_csv(f'{S3_PATH}analysis/csv/2010_place_covs.csv', index=False)

## Get UA Covs 
juris = pd.read_csv(f'{S3_PATH}analysis/csv/2010_place_covs.csv')
juris = juris.drop(columns=['GISJOIN_PLACE','source'])

uas = juris[['UACODE','pop_total']].groupby('UACODE').sum().reset_index().rename(columns={'pop_total':'ua_pop'})
juris = juris.merge(uas, how='left',on='UACODE')

juris['weight'] = juris['pop_total']/juris['ua_pop']

columns = list(juris.columns)
remove_column = ['UACODE','weight','pop_total','house_agg','rent_agg']
for col in remove_column:
    columns.remove(col)
for column in columns:
    juris[column] = juris[column]*juris['weight']

juris = juris.groupby(['UACODE']).sum().reset_index()

juris = juris.drop(columns=['weight'])
juris = juris[juris['pop_total']!=0]

juris.to_csv(f'{S3_PATH}analysis/csv/2010_ua_covs.csv', index=False)

## Get Regulations
nlp = pd.read_csv(f'{S3_PATH}nlp/csv/nlp_final.csv')
nlp = nlp[(nlp['urbanized2010']==1) & (nlp['urbanized2018']==1)]

old = nlp[(nlp['year']>=2005) & (nlp['year']<=2010)]
old = old.sort_values(['year','month','day'], ascending=False).drop_duplicates(subset='gisjoin')
new = nlp[(nlp['year']>=2010) & (nlp['year']<=2015)]
new = new.sort_values(['year','month','day'], ascending=True).drop_duplicates(subset='gisjoin')

nlp = pd.concat([old,new]).reset_index(drop=True)
weights = pd.read_csv(f'{S3_PATH}gis/csv/2010_juris_weights.csv').rename(columns={'GISJOIN_PLACE':'gisjoin'})

nlp = weights.merge(nlp, how='left', on='gisjoin')[['gisjoin','frac','wrluri','alpha', 'beta', 'tau']].rename(columns={'gisjoin':'GISJOIN_PLACE'})

flag = nlp
flag['flag'] = flag['wrluri'].apply(lambda x: 0 if math.isnan(x) else 1)
flag = flag[['GISJOIN_PLACE','flag']]

# Get Inverse Mills Ratio, Obtain New Weights # Sample Selection Bias and IV GO BACK TO Section 19.6.2 in the second edition of "Econometric Analysis of Cross Section and Panel Data,"
data = pd.read_csv(f'{S3_PATH}analysis/csv/2010_place_covs.csv')
data = data.merge(flag, how='inner', on='GISJOIN_PLACE').replace([np.inf, -np.inf], np.nan).dropna()

remove_list = ['GISJOIN_PLACE','UACODE','frac','source','wrluri', 'alpha', 'beta', 'tau', 'flag']

y = data['flag']
X = data[[i for i in list(data.columns) if i not in remove_list]]
X = sm.add_constant(X)

model = Probit(y, X, missing='drop').fit()
print(model.summary())
data['mr'] = model.predict(X)
data['imr'] = data['mr'].apply(lambda x: 1/x)

data = data.merge(nlp, how='inner', on='GISJOIN_PLACE')

data = data.drop(columns=['flag_y']).rename(columns={'flag_x':'flag'})
data.to_csv(f'{S3_PATH}analysis/csv/2010_place_heckman.csv',index=False)

data = data[['GISJOIN_PLACE','UACODE','frac','wrluri', 'alpha', 'beta', 'tau', 'imr']]

data['weight'] = data['imr']*data['frac']

data = data[data['wrluri'].notna()]
uas = data[['UACODE','weight']].groupby('UACODE').sum().reset_index().rename(columns={'weight':'ua_weight'})
data = data.merge(uas, how='left', on='UACODE')
data['weight'] = data['weight']/data['ua_weight']

indices = ['wrluri','alpha','beta','tau']
for index in indices:
    data[index] = data['weight']*data[index]

uas = data[['UACODE','wrluri','alpha','beta','tau','frac']].groupby('UACODE').sum().reset_index()

uas.to_csv(f'{S3_PATH}analysis/csv/final_indices.csv', index=False)

# Merge Tract Covs w/ Gini and Surrogate Index
reg = pd.read_csv(f'{S3_PATH}analysis/csv/final_indices.csv')
gini = pd.read_csv(f'{S3_PATH}gis/csv/2010_gini.csv')

uas_covs = pd.read_csv(f'{S3_PATH}analysis/csv/2010_ua_covs.csv')
uas_covs = uas_covs.drop(columns=['ua_pop'])
colum = list(uas_covs.columns)
colum.remove('UACODE')
colum = [f'{col}_metro' for col in colum]
colum.insert(0, 'UACODE')
uas_covs.columns = colum

uas = uas_covs.merge(gini, how='left', on='UACODE').merge(reg, how='left', on='UACODE')
uas.to_csv(f'{S3_PATH}analysis/csv/2010_uas_heckman.csv')

surr = pd.read_csv(f'{S3_PATH}opp_atlas/csv/surrogates_2010.csv')
tracts = surr.merge(uas, how='left', on='UACODE')

tracts.to_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv', index=False)

# 3rd Stage Covs
# Public Expenditures (Place Level)
govs = pd.read_csv(f'{S3_PATH}analysis/csv/2010_govt_fin/2010_govt_fin.txt')[['ID','Name','Population','Total Revenue','Total Expenditure']]
govs['ID'] = govs['ID'].astype(int)

key = pd.read_fwf('{S3_PATH}analysis/csv/2010_govt_fin/Fin_GID_2010.txt')
key.columns = ['id_name','name','FIPS','code','impute']
key['ID'] = key['id_name'].apply(lambda x: int(x[:9]))

govs = govs.merge(key, how='left', on='ID')
govs = govs[govs['FIPS'].notna()]
govs['FIPS'] = govs['FIPS'].astype(int).astype(str)
govs = govs[govs['FIPS'].str.len()>5]
govs['expen_per_cap'] = govs.apply(lambda x: x['Total Expenditure']/x['Population'], axis=1)

counties = govs[govs['Name'].str.contains('COUNTY')]
counties['PLACE_CODE'] = counties['FIPS'].apply(lambda x: x[-3:])
counties['STATE_CODE'] = counties['FIPS'].apply(lambda x: x[:1] if len(x)==9 else (x[:2]))
counties['GISJOIN_PLACE'] = counties.apply(lambda x: f"G{x['STATE_CODE'].zfill(2)}0{x['PLACE_CODE']}0", axis=1)

places = govs[~govs['Name'].str.contains('COUNTY')]
places['PLACE_CODE'] = places['FIPS'].apply(lambda x: x[-5:])
places['STATE_CODE'] = places['FIPS'].apply(lambda x: x[:1] if len(x)==9 else (x[:2]))
places['GISJOIN_PLACE'] = places.apply(lambda x: f"G{x['STATE_CODE'].zfill(2)}0{x['PLACE_CODE'].zfill(5)}", axis=1)

govs = pd.concat([places,counties])[['GISJOIN_PLACE','expen_per_cap']]

govs.to_csv(f'{S3_PATH}analysis/csv/2010_govt_fin/final.csv', index=False)

tracts = pd.read_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv')
govs = pd.read_csv(f'{S3_PATH}analysis/csv/2010_govt_fin/final.csv')

tracts = pd.merge(tracts, govs, how='left', on='GISJOIN_PLACE')
tracts.to_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv', index=False)

# Real Estate (Metro Level)
rent_tract = pd.read_csv(f'{S3_PATH}analysis/csv/2010_tract_rents/nhgis0058_ds176_20105_tract.csv', encoding='latin-1')

def rent_expectation(x, alpha):
    # alpha is a tuning parameter for logistic function through which `diff` is transformed. Recommended setting ~3
    ranges = [(0,200),(200,299),(300,499),(500,749),(750,1000),(1000,3000)]

    pop = x['JS9E021']
    if pop==0:
        return 0

    expec = 0
    for i in list(range(22,28)):
        pop_dup = pop + x[f'JS9E0{i}']
        prob = x[f'JS9E0{i}'] / pop
        min_share = 0 
        for min in list(range(22,int(i)+1)):
            min_share += x[f'JS9E0{min}']/pop_dup
        max_share = 0 
        for max in list(range( int(i), 28)):
            if len(str(max)) == 1:
                max = f'0{max}'
            max_share += x[f'JS9E0{max}']/pop_dup
        diff = max_share - min_share
        diff = (2/(1+math.exp(-alpha*diff)))-1
        
        value = (ranges[int(i)-22][1]+ranges[int(i)-22][0])/2 + diff*(ranges[int(i)-22][1]-ranges[int(i)-22][0])/2

        expec += value*prob

    return expec

for index, row in tqdm(rent_tract.iterrows(), total=len(rent_tract)):
    rent_tract.at[index, "avg_rent_2_bed"] = rent_expectation(row, 3)

rent_tract.to_csv(f'{S3_PATH}analysis/csv/2010_tract_rents/2010_tract_rents.csv', index=False)

rent_tract = pd.read_csv(f'{S3_PATH}analysis/csv/2010_tract_rents/2010_tract_rents.csv')
rent_tract = rent_tract[['GISJOIN','avg_rent_2_bed']].rename(columns={'GISJOIN':'GISJOIN_TRACT'})

tracts = pd.read_csv(f'{S3_PATH}opp_atlas/csv/surrogates_2010.csv')[['GISJOIN_TRACT','UACODE','p_p_p','pop_total']].drop_duplicates()
uas = tracts[['UACODE','pop_total']].groupby('UACODE').sum().reset_index().rename(columns={'pop_total':'ua_pop'})
tracts = tracts.merge(uas,how='left',on='UACODE')
tracts['weight'] = tracts['pop_total']/tracts['ua_pop']

tracts = tracts.merge(rent_tract, how='left',on='GISJOIN_TRACT')
tracts = tracts[(tracts['avg_rent_2_bed']!=0) & (tracts['weight']!=0)]
tracts['opp_rent'] = (tracts['p_p_p']/tracts['avg_rent_2_bed'])*tracts['weight']
tracts = tracts[['UACODE','opp_rent']].groupby('UACODE').sum().reset_index()

data = pd.read_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv')
data = data.merge(tracts, how='left', on='UACODE')

data.to_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv', index=False)

# Create Reg-Sample
data = pd.read_csv(f'{S3_PATH}analysis/csv/2010_tract-place_covs_reg_gini.csv')
data = data.dropna()

# Prepare Columns for Stata
stata_cols = data.columns
stata_cols = [f'd{col}'.replace('>','great').replace('<','less').replace('-','to') if col[0].isnumeric() else f'{col}'.replace('>','great').replace('<','less').replace('-','to') for col in stata_cols]
stata_cols = ['s_par_perc_fam_met' if col=='single_parent_perc_families_metro' else col for col in stata_cols]
stata_cols = ['s_f_h_att_met' if col=='single_family_home_attach_perc_metro' else col for col in stata_cols]
stata_cols = ['gini' if col=='Gini' else col for col in stata_cols]
data.columns = stata_cols

# Transformations
data['gini_sq'] = data['gini'].apply(lambda x: x**2)
data['pop_total'] = data['pop_total'].apply(lambda x: math.log(x) if x!=0 else 0)
data['pop_total_metro'] = data['pop_total_metro'].apply(lambda x: math.log(x) if x!=0 else 0)
data['greater_college_perc'] = data['college_perc']+data['grad_perc']
data['greater_college_perc_metro'] = data['college_perc_metro']+data['grad_perc_metro']
data['p_p_p'] = data['p_p_p']/10000
data['p_p_25'] = data['p_p_25']/10000
data['p_p_50'] = data['p_p_50']/10000
data['p_p_75'] = data['p_p_75']/10000

data.to_csv(f'{S3_PATH}analysis/csv/reg_sample.csv', index=False)

################################ 2SLS ######################################
data = pd.read_csv(f'{S3_PATH}analysis/csv/reg_sample.csv')

labels = '''
lab var p_p_p "Pooled"
lab var p_p_25 "25th Pctl (\\$10k)"
lab var p_p_50 "50th Pctl (\\$10k)"
lab var p_p_75 "75th Pctl (\\$10k)"

lab var wrluri "WRLURI"
lab var alpha "\\alpha"
lab var beta "\\beta"
lab var tau "\\tau"

lab var gini "Gini"
lab var gini_sq "Gini Sq."

lab var pop_total "Ln Total Pop."
lab var female_perc "Female (\\%)"
lab var white_perc "White (\\%)"
lab var black_perc "Black (\\%)"
lab var asian_perc "Asian (\\%)"
lab var hisp_perc "Hispanic (\\%)" 
lab var great65_perc "$>$65 y.o. (\\%)" 
lab var less18_perc "$<$18 y.o. (\\%)"
lab var married_perc "Married HH (\\%)"
lab var greater_college_perc "Grad. or Bachelors (\\%)"
lab var married_families_perc "Married Families (\\%)" 
lab var single_parent_perc "Single Parent HH (\\%)"
lab var single_parent_perc_families "Single Parent Families (\\%)"
lab var singles_perc "Single HH (\\%)"
lab var own_child_perc "Child w/ Parents (\\%)"
lab var relative_child_perc "Child w/ Relatives (\\%)"
lab var foster_child_perc "Child in Foster (\\%)"
lab var great1hr_comm_perc "$>$1hr Commute (\\%)"
lab var less20_comm_perc "$<$20m Commute (\\%)"
lab var private_school_perc "Private Prim./Sec. (\\%)"
lab var grad_perc "Grad. Deg. (\\%)"
lab var college_perc "Bachelors Deg. (\\%)"
lab var some_college_perc "Some College (\\%)"
lab var hs_drop_perc "HS Drop (\\%)"
lab var less15k_perc "$<$\\\\$15k (\\%)"
lab var d35to75k_perc "\\\\$35-75k (\\%)" 
lab var great125k_perc "$>$\\\\$125k (\\%)" 
lab var public_assist_perc "On Public Asst. (\\%)"
lab var perc_fam_poor "Poor Families (\\%)"
lab var foreign_born_perc "Foreign Born (\\%)"

lab var pop_total_metro "Ln Total Pop."
lab var female_perc_metro "Female (\\%)"
lab var white_perc_metro "White (\\%)"
lab var black_perc_metro "Black (\\%)"
lab var asian_perc_metro "Asian (\\%)"
lab var hisp_perc_metro "Hispanic (\\%)"
lab var less18_perc_metro "$<$18 y.o. (\\%)"
lab var s_par_perc_fam_met "Single Parent HH (\\%)"
lab var own_child_perc_metro "Child w/ Parents (\\%)"
lab var great1hr_comm_perc_metro "$>$1hr Commute (\\%)"
lab var less20_comm_perc_metro "$<$20m Commute (\\%)"
lab var greater_college_perc_metro "Grad. or Bachelors (\\%)"
lab var hs_drop_perc_metro "HS Drop (\\%)"
lab var less15k_perc_metro "$<$\\$15k (\\%)"
lab var d35to75k_perc_metro "\\$35-75k (\\%)"
lab var  great125k_perc_metro  "$>$\\$125k (\\%)"
lab var perc_fam_poor_metro "Poor Families (\\%)"
lab var foreign_born_perc_metro "Foreign Born (\\%)"

lab var single_family_home_perc_metro  "Family H (\\% UA)"
lab var opp_rent "Ad.Inc./Rent (\\$, UA)"
lab var expen_per_cap "Expenditures per Cap (P)"
'''

local_covs = ['pop_total', 'female_perc', 'white_perc', 'black_perc', 'asian_perc', 'hisp_perc', 'less18_perc', 'single_parent_perc_families', 'own_child_perc', 'great1hr_comm_perc', 'less20_comm_perc', 'greater_college_perc', 'hs_drop_perc', 'less15k_perc', 'd35to75k_perc', ' great125k_perc', 'perc_fam_poor', 'foreign_born_perc']

stata.pdataframe_to_data(data, force=True)
stata.run(f'''

{labels}

local local_covs {str(local_covs).replace('[','').replace(']','').replace(',','').replace("'",'')}

estpost summarize `local_covs'

esttab using "sum_stat/local_covs.tex", long booktabs replace label ///
    cells("mean(fmt(3)) sd(fmt(3)) min(fmt(3)) max(fmt(3))") nomtitle nonumber title(Tract-Level Covariates)


*************** JUST IDENTIFIED
eststo: ivreg2 p_p_p (wrluri=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (wrluri=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (wrluri=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (wrluri=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/ji_wrluri.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Just Identified): WRLURI (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (alpha=gini) `local_covs', robust first ffirst savefirst savefprefix(first_)
eststo: ivreg2 p_p_25 (alpha=gini) `local_covs', robust first ffirst savefirst savefprefix(first_)
eststo: ivreg2 p_p_50 (alpha=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (alpha=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/ji_alpha.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Just Identified): $\\alpha$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (beta=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (beta=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (beta=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (beta=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/ji_beta.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Just Identified): $\\beta$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (tau=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (tau=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (tau=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (tau=gini) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/ji_tau.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Just Identified): $\\tau$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)"))

eststo clear

esttab first* using "regressions/ji_first.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Just Identified): First Stage") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N, fmt(%4.0f) label("Observations"))

eststo clear

*************** OVER IDENTIFIED

eststo: ivreg2 p_p_p (wrluri=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (wrluri=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (wrluri=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (wrluri=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/oi_wrluri.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Overdentified): WRLURI (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat j, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)" "Hansen (J-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (alpha=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (alpha=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (alpha=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (alpha=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/oi_alpha.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Overdentified): $\\alpha$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat j, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)" "Hansen (J-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (beta=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (beta=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (beta=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (beta=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/oi_beta.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Overdentified): $\\beta$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat j, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)" "Hansen (J-Stat)"))

eststo clear

eststo: ivreg2 p_p_p (tau=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_25 (tau=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_50 (tau=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 
eststo: ivreg2 p_p_75 (tau=gini gini_sq) `local_covs', robust first ffirst savefirst savefprefix(first_) 

esttab using "regressions/oi_tau.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Overdentified): $\\tau$ (Tracts)") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a widstat j, fmt(%4.0f %4.3f %4.3f %4.3f) label("Observations" "Adj. R^2" "Cragg-Donald (F-Stat)" "Hansen (J-Stat)"))

eststo clear

esttab first* using "regressions/oi_first.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("2SLS (Overdentified): First Stage") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N, fmt(%4.0f) label("Observations"))
''')

################################ GSEM ######################################
local_covs = ['pop_total', 'female_perc', 'white_perc', 'black_perc', 'asian_perc', 'hisp_perc', 'less18_perc', 'single_parent_perc_families', 'own_child_perc', 'great1hr_comm_perc', 'less20_comm_perc', 'greater_college_perc', 'hs_drop_perc', 'less15k_perc', 'd35to75k_perc', ' great125k_perc', 'perc_fam_poor', 'foreign_born_perc']
metro_covs = ['pop_total_metro', 'female_perc_metro', 'white_perc_metro', 'black_perc_metro', 'asian_perc_metro', 'hisp_perc_metro', 'less18_perc_metro', 's_par_perc_fam_met', 'own_child_perc_metro', 'great1hr_comm_perc_metro', 'less20_comm_perc_metro', 'greater_college_perc_metro', 'hs_drop_perc_metro', 'less15k_perc_metro', 'd35to75k_perc_metro', ' great125k_perc_metro', 'perc_fam_poor_metro', 'foreign_born_perc_metro']

stata.pdataframe_to_data(data, force=True)
regs = ['wrluri', 'alpha', 'beta', 'tau']
for reg in regs:
    stata.run(f'''

        {labels}

        local local_covs {str(local_covs).replace('[','').replace(']','').replace(',','').replace("'",'')}
        local metro_covs {str(metro_covs).replace('[','').replace(']','').replace(',','').replace("'",'')}

        eststo: ivreg2 opp_rent ({reg}=gini) `local_covs', robust
        predict opp_rent_hat_{reg}
        eststo: ivreg2 expen_per_cap ({reg}=gini) `local_covs', robust
        predict expen_per_cap_hat_{reg}
        eststo: ivreg2 single_family_home_perc_metro ({reg}=gini) `local_covs', robust
        predict sfh_metro_hat_{reg}

        esttab using "regressions/gsem_second_{reg}.tex", replace ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        title("GSEM: 2nd Stage Given \\{reg}") ///
        label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
        stats(N, fmt(%4.0f) label("Observations"))

        eststo clear 

        eststo: reg p_p_p opp_rent_hat_{reg} expen_per_cap_hat_{reg} sfh_metro_hat_{reg} pop_total female_perc white_perc black_perc asian_perc hisp_perc less18_perc single_parent_perc_families own_child_perc great1hr_comm_perc less20_comm_perc greater_college_perc hs_drop_perc less15k_perc d35to75k_perc great125k_perc, robust
        eststo: reg p_p_25 opp_rent_hat_{reg} expen_per_cap_hat_{reg} sfh_metro_hat_{reg} pop_total female_perc white_perc black_perc asian_perc hisp_perc less18_perc single_parent_perc_families own_child_perc great1hr_comm_perc less20_comm_perc greater_college_perc hs_drop_perc less15k_perc d35to75k_perc great125k_perc, robust
        eststo: reg p_p_50 opp_rent_hat_{reg} expen_per_cap_hat_{reg} sfh_metro_hat_{reg} pop_total female_perc white_perc black_perc asian_perc hisp_perc less18_perc single_parent_perc_families own_child_perc great1hr_comm_perc less20_comm_perc greater_college_perc hs_drop_perc less15k_perc d35to75k_perc great125k_perc, robust
        eststo: reg p_p_75 opp_rent_hat_{reg} expen_per_cap_hat_{reg} sfh_metro_hat_{reg} pop_total female_perc white_perc black_perc asian_perc hisp_perc less18_perc single_parent_perc_families own_child_perc great1hr_comm_perc less20_comm_perc greater_college_perc hs_drop_perc less15k_perc d35to75k_perc great125k_perc, robust

        esttab using "regressions/gsem_third_{reg}.tex", replace ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        title("GSEM: 3rd Stage Given \\{reg}") ///
        label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
        stats(N, fmt(%4.0f) label("Observations"))

        eststo clear
        est clear 
    
    ''')

################################ HECKMAN ######################################
place = pd.read_csv(f'{S3_PATH}analysis/csv/2010_place_heckman.csv')

# Transformations
place['pop_total'] = place['pop_total'].apply(lambda x: math.log(x) if x!=0 else 0)
place['source'] = (place['source']=='counties')*1
place['greater_coll'] = place['grad_perc']+place['college_perc']
place['home_owners_perc'] = place['white_owner_perc']+place['black_owner_perc']+place['asian_owner_perc']
place = pd.merge(place, place[['UACODE','white_perc']].groupby('UACODE').mean().reset_index().rename(columns={'white_perc':'metro_white_perc'}), how='left', on='UACODE')
place['whiter_flag'] = (place['white_perc']>place['metro_white_perc'])*1

# Prepare Columns for Stata
stata_cols = place.columns
stata_cols = [f'D{col}'.replace('>','great').replace('<','less').replace('-','to') if col[0].isnumeric() else f'{col}'.replace('>','great').replace('<','less').replace('-','to') for col in stata_cols]
place.columns = stata_cols
drop_list = ['mr','imr','frac','UACODE','metro_white_perc']
place = place.drop(columns=drop_list)

# Define Variable Set
# total_covs = ['pop_total', 'source', 'female_perc', 'white_perc', 'black_perc', 'asian_perc', 'hisp_perc', 'great65_perc', 'less18_perc', 'married_perc', 'married_families_perc', 'single_parent_perc', 'single_parent_perc_families', 'singles_perc', 'own_child_perc', 'relative_child_perc', 'foster_child_perc', 'vacant_houses_perc', 'owner_houses_perc', 'white_owner_perc', 'black_owner_perc', 'asian_owner_perc', 'white_renter_perc', 'black_renter_perc', 'asian_renter_perc', 'less15k_houses_perc', 'great500k_houses_perc', 'D15to45k_houses_perc', 'D150to500k_houses_perc', 'house_median', 'house_agg', 'perc_owner_white', 'perc_owner_black', 'perc_owner_asian', 'less100_rent_perc', 'great1000_rent_perc', 'D100to400_rent_perc', 'D500to1000_rent_perc', 'rent_median', 'rent_agg', 'single_family_home_perc', 'single_family_home_attach_perc', 'D2to9_units_perc', 'D10to50_units_perc', 'mobile_units_perc', 'great1hr_comm_perc', 'less20_comm_perc', 'private_school_perc', 'grad_perc', 'college_perc', 'some_college_perc', 'hs_drop_perc', 'less15k_perc', 'D35to75k_perc', 'great125k_perc', 'public_assist_perc', 'perc_fam_poor', 'foreign_born_perc']
covs = ['pop_total', 'source', 'great65_perc', 'less18_perc', 'white_perc', 'black_perc', 'asian_perc', 'hisp_perc', 'whiter_flag', 'married_families_perc', 'singles_perc', 'home_owners_perc', 'great1hr_comm_perc', 'less20_comm_perc', 'greater_coll', 'less15k_perc', 'D35to75k_perc', 'great125k_perc','foreign_born_perc']

labels = '''
lab var wrluri "WRLURI"
lab var alpha "\\alpha"
lab var beta "\\beta"
lab var tau "\\tau"

lab var flag "Sample Flag"

lab var pop_total "Ln Total Pop. (P)"
lab var source "County Flag"
lab var female_perc "Female (\% P)"
lab var white_perc "White (\% P)"
lab var black_perc "Black (\% P)"
lab var asian_perc "Asian (\% P)"
lab var whiter_flag "Whiter than UA (P)"
lab var hisp_perc "Hispanic (\% P)" 
lab var home_owners_perc "H.owners (\% P)"
lab var great65_perc "$>$65 y.o. (\% P)" 
lab var less18_perc "$>$18 y.o. (\% P)"
lab var married_perc "Married HH (\% P)"
lab var married_families_perc "Married Families (\% P)" 
lab var single_parent_perc "Single Parent HH (\% P)"
lab var single_parent_perc_families "Single Parent Families (\% P)"
lab var singles_perc "Single HH (\% P)"
lab var own_child_perc "Child w/ Parents (\% P)"
lab var relative_child_perc "Child w/ Relatives (\% P)"
lab var foster_child_perc "Child in Foster (\% P)"
lab var great1hr_comm_perc "$>$1hr Commute (\% P)"
lab var less20_comm_perc "$>$20m Commute (\% P)"
lab var private_school_perc "Private Prim./Sec. (\% P)"
lab var greater_coll "Bachelors or Grad (\% P)"
lab var grad_perc "Grad. Deg. (\% P)"
lab var college_perc "Bachelors Deg. (\% P)"
lab var some_college_perc "Some College (\% P)"
lab var hs_drop_perc "HS Drop (\% P)"
lab var less15k_perc "$<$\$15k (1990 \$) (\% P)"
lab var D35to75k_perc "\$35-75k (1990 \$) (\% P)" 
lab var great125k_perc "$>$125k (1990 \$) (\% P)" 
lab var public_assist_perc "On Public Asst. (\% P)"
lab var perc_fam_poor "Poor Families (\% P)"
lab var foreign_born_perc "Foreign Born (\% P)"
lab var vacant_houses_perc "Vacant Houses (\% P)"
lab var owner_houses_perc "Owner Occupied H (\% P)"
lab var white_owner_perc "White H.owner (\% P)"
lab var black_owner_perc "Black H.owner (\% P)"
lab var asian_owner_perc "Asian H.owner (\% P)"
lab var white_renter_perc "White Renter (\% P)"
lab var black_renter_perc "Black Renter (\% P)"
lab var asian_renter_perc "Asian Renter (\% P)"
lab var less15k_houses_perc "$>$\$15k H Value (1990 \$) (\% P)" 
lab var great500k_houses_perc "$>$\$500k H Value (1990 \$) (\% P)"
lab var D15to45k_houses_perc "\$15k-\$45k H Value (1990 \$) (\% P)"
lab var D150to500k_houses_perc "\$150k-\$500k H Value (1990 \$) (\% P)"
lab var house_median "H Median Value (1990 \$) (\% P)"
lab var house_agg "H Agg. Value (1990 \$) (\% P)"
lab var perc_owner_white "H.owners White (\% P)"
lab var perc_owner_black "H.owners Black (\% P)"
lab var perc_owner_asian "H.owners Asian (\% P)"
lab var less100_rent_perc "$>$\$100 Rent (1990 \$) (\% P)"
lab var great1000_rent_perc "$>$\$1000 Rent (1990 \$) (\% P)"
lab var D100to400_rent_perc "\$100-\$400 Rent (1990 \$) (\% P)"
lab var D500to1000_rent_perc "\$500-\$1000 Rent(1990 \$) (\% P)"
lab var rent_median "Rent Median (1990 \$) (\% P)"
lab var rent_agg "Rent Agg. (1990 \$) (\% P)"
lab var single_family_home_perc "Single Family H (\% P)"
lab var single_family_home_attach_perc "Single Fam (Attach) (\% P)"
lab var D2to9_units_perc "2-9 Units (\% P)"
lab var D10to50_units_perc "10-50 Units (\% P)"
lab var mobile_units_perc "Mobile Homes (\% P)"
'''

stata.pdataframe_to_data(place, force=True)
stata.run(f'''
est clear

{labels}

local covs {str(covs).replace('[','').replace(']','').replace(',','').replace("'",'')}

eststo: heckman wrluri `covs', select(flag = `covs') robust
eststo: heckman alpha `covs', select(flag = `covs') robust
eststo: heckman beta `covs', select(flag = `covs') robust
eststo: heckman tau `covs', select(flag = `covs') robust

esttab using "regressions/place_heckman.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("Heckman: Regulatory Indices") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none)''')

################################ DETERMINANTS MOBILITY ######################################
covariates = pd.read_csv(f'{S3_PATH}opp_atlas/csv/1990_tract_covs_final.csv')
outcomes = pd.read_csv(f'{S3_PATH}opp_atlas/csv/tract_outcomes/merged.csv')
outcomes = outcomes.rename(columns = {'tract':'2010_TRACT_GID'})
covs_out = outcomes.merge(covariates, how='left', on='2010_TRACT_GID').dropna()

# Prepare Columns for Stata
stata_cols = covs_out.columns
stata_cols = [f'D{col}'.replace('>','great').replace('<','less').replace('-','to') if col[0].isnumeric() else f'{col}'.replace('>','great').replace('<','less').replace('-','to') for col in stata_cols]
covs_out.columns = stata_cols

# Transformations
covs_out['pop_total'] = covs_out['pop_total'].apply(lambda x: math.log(x) if x!=0 else 0)

# Define Variable Set
covs = ['pop_total', 'female_perc', 'white_perc', 'black_perc', 'asian_perc', 'hisp_perc', 'great65_perc', 'less18_perc', 'married_perc', 'married_families_perc', 'single_parent_perc', 'single_parent_perc_families', 'singles_perc', 'own_child_perc', 'relative_child_perc', 'foster_child_perc', 'vacant_houses_perc', 'owner_houses_perc', 'white_owner_perc', 'black_owner_perc', 'asian_owner_perc', 'white_renter_perc', 'black_renter_perc', 'asian_renter_perc', 'less15k_houses_perc', 'great500k_houses_perc', 'D15to45k_houses_perc', 'D150to500k_houses_perc', 'house_median', 'house_agg', 'perc_owner_white', 'perc_owner_black', 'perc_owner_asian', 'less100_rent_perc', 'great1000_rent_perc', 'D100to400_rent_perc', 'D500to1000_rent_perc', 'rent_median', 'rent_agg', 'single_family_home_perc', 'single_family_home_attach_perc', 'D2to9_units_perc', 'D10to50_units_perc', 'mobile_units_perc', 'great1hr_comm_perc', 'less20_comm_perc', 'private_school_perc', 'grad_perc', 'college_perc', 'some_college_perc', 'hs_drop_perc', 'less15k_perc', 'D35to75k_perc', 'great125k_perc', 'public_assist_perc', 'perc_fam_poor', 'foreign_born_perc']

labels = '''

lab var p_p_p "Pooled (2015 \$)"
lab var p_p_25 "25th \\% (2015 \$)"
lab var p_p_50 "50th \\% (2015 \$)"
lab var p_p_75 "75th \\% (2015 \$)"

lab var pop_total "Ln Total Pop. (T)"
lab var female_perc "Female (\\%)"
lab var white_perc "White (\\%)"
lab var black_perc "Black (\\%)"
lab var asian_perc "Asian (\\%)"
lab var hisp_perc "Hispanic (\\%)" 
lab var great65_perc "$>$65 y.o. (\\%)" 
lab var less18_perc "$<$18 y.o. (\\%)"
lab var married_perc "Married HH (\\%)"
lab var married_families_perc "Married Families (\\%)" 
lab var single_parent_perc "Single Parent HH (\\%)"
lab var single_parent_perc_families "Single Parent Families (\\%)"
lab var singles_perc "Single HH (\\%)"
lab var own_child_perc "Child w/ Parents (\\%)"
lab var relative_child_perc "Child w/ Relatives (\\%)"
lab var foster_child_perc "Child in Foster (\\%)"
lab var great1hr_comm_perc "$>$1hr Commute (\\%)"
lab var less20_comm_perc "$<$20m Commute (\\%)"
lab var private_school_perc "Private Prim./Sec. (\\%)"
lab var grad_perc "Grad. Deg. (\\%)"
lab var college_perc "Bachelors Deg. (\\%)"
lab var some_college_perc "Some College (\\%)"
lab var hs_drop_perc "HS Drop (\\%)"
lab var less15k_perc "$<$\$15k (\\%)"
lab var D35to75k_perc "\$35-75k (\\%)" 
lab var great125k_perc "$>$125k (\\%)" 
lab var public_assist_perc "On Public Asst. (\\%)"
lab var perc_fam_poor "Poor Families (\\%)"
lab var foreign_born_perc "Foreign Born (\\%)"
lab var vacant_houses_perc "Vacant Houses (\\%)"
lab var owner_houses_perc "Owner Occupied H (\\%)"
lab var white_owner_perc "White H.owner (\\%)"
lab var black_owner_perc "Black H.owner (\\%)"
lab var asian_owner_perc "Asian H.owner (\\%)"
lab var white_renter_perc "White Renter (\\%)"
lab var black_renter_perc "Black Renter (\\%)"
lab var asian_renter_perc "Asian Renter (\\%)"
lab var less15k_houses_perc "$<$\$15k H Value (\\%)" 
lab var great500k_houses_perc "$>$\$500k H Value (\\%)"
lab var D15to45k_houses_perc "\$15k-\$45k H Value (\\%)"
lab var D150to500k_houses_perc "\$150k-\$500k H Value (\\%)"
lab var house_median "H Median Value (\\%)"
lab var house_agg "H Agg. Value (\\%)"
lab var perc_owner_white "H.owners White (\\%)"
lab var perc_owner_black "H.owners Black (\\%)"
lab var perc_owner_asian "H.owners Asian (\\%)"
lab var less100_rent_perc "$<$\$100 Rent (\\%)"
lab var great1000_rent_perc "$>$\$1000 Rent (\\%)"
lab var D100to400_rent_perc "\$100-\$400 Rent (\\%)"
lab var D500to1000_rent_perc "\$500-\$1000 Rent(\\%)"
lab var rent_median "Rent Median (\\%)"
lab var rent_agg "Rent Agg. (\\%)"
lab var single_family_home_perc "Single Family H (\\%)"
lab var single_family_home_attach_perc "Single Fam (Attach) (\\%)"
lab var D2to9_units_perc "2-9 Units (\\%)"
lab var D10to50_units_perc "10-50 Units (\\%)"
lab var mobile_units_perc "Mobile Homes (\\%)"
'''

stata.pdataframe_to_data(covs_out, force=True)
stata.run(f'''

{labels}

local covs {str(covs).replace('[','').replace(']','').replace(',','').replace("'",'')}

eststo: reg p_p_p `covs', robust
eststo: reg p_p_25 `covs', robust
eststo: reg p_p_50 `covs', robust
eststo: reg p_p_75 `covs', robust

esttab using "regressions/det_upward_mobility.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("OLS: Income at 35") ///
label long booktabs alignment(D{{.}}{{.}}{{-1}}) collabels(none) ///
stats(N r2_a, fmt(%4.0f %4.3f %4.3f) label("Observations" "Adj. R^2"))
''')

##################################################### SUMMARY STATISTICS ######################################################
uas = pd.read_csv(f'{S3_PATH}analysis/csv/final_indices.csv')
uas.columns = ['UACODE','WRLURI','\\alpha','\\beta','\\tau','Frac. Land']
#1
uas[['WRLURI','\\alpha','\\beta','\\tau','Frac. Land']].describe().round(decimals=3).to_latex('sum_stat/regulatory_indices.tex', caption='Regulatory Indices Summary')
#2 
uas[['WRLURI',f'\\alpha',f'\\beta',f'\\tau']].corr().round(decimals=3).to_latex('sum_stat/regulatory_correlations.tex', caption='Regulatory Indices Correlations')

auto_corrs = pd.read_csv(f'{S3_PATH}nlp/csv/auto_corrs.csv').round(decimals=3)
auto_corrs.columns = ['Lag','WRLURI','\\alpha','\\beta','\\tau','Obs.']
auto_corrs.to_latex('sum_stat/auto_corrs.tex', caption='Regulatory Indices Autocorrelations', index=False)

reg = pd.read_csv(f'{S3_PATH}analysis/csv/reg_sample.csv')
ops = reg[['p_p_p','p_p_25','p_p_50','p_p_75']]
ops.columns = ['Unconditional','25th%','50th%','75th%',]
ops.describe().round(decimals=3).to_latex('sum_stat/opps.tex', caption='Surrogate Indices')

########################### EXPORT
S3FS.put('regressions',f'{S3_PATH}outputs/regressions',recursive=True)
S3FS.put('sum_stat',f'{S3_PATH}outputs/sum_stat',recursive=True)