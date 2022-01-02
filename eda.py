from os import remove
import pandas as pd
from global_var import ROOT_DIR
import numpy as np
from matplotlib import pyplot as plt
import stata_setup
stata_setup.config('/Applications/Stata/', 'be')
from pystata import stata

######## Build tract outcomes and covariates with MSA
# cz = pd.read_csv(f'{ROOT_DIR}/1data/csv/tract_outcomes_simple.csv')
# cz = cz[['state','county','cz','czname','tract','kfr_pooled_pooled_p25']]
# cz_covs = pd.read_csv(f'{ROOT_DIR}/1data/csv/tract_covariates.csv')
# cz = pd.merge(cz, cz_covs, how='left', on=['state','county','cz','czname','tract'])
# cz_metro = pd.read_csv(f'{ROOT_DIR}/1data/csv/czlma903.csv')
# cz_metro = cz_metro[['CZ90','MSA 1993']].rename(columns={'CZ90':'cz', 'MSA 1993':'msa'}).drop_duplicates()
# cz = pd.merge(cz, cz_metro, how='left', on='cz')
# cz = cz[(cz['msa']!=0) & (cz['msa'].notna())]

######## Build Census Places Population data with MSA
sub_est = pd.read_csv(f'{ROOT_DIR}/1data/csv/sub-est00int.csv', encoding = "ISO-8859-1")
sub_est = sub_est[ (sub_est['SUMLEV']!=50) & (sub_est['SUMLEV']!=40)]
sub_est['COUNTY'] = sub_est.apply(lambda x:  str(x['STATE'])+'0'+str(x['COUNTY']) if len(str(x['COUNTY']))==2 else (str(x['STATE'])+'00'+str(x['COUNTY']) if len(str(x['COUNTY']))==1 else str(x['STATE'])+str(x['COUNTY'])), axis = 1).astype(int)
sub_est = sub_est[['SUMLEV','PLACE','COUSUB','STATE','STNAME','COUNTY','POPESTIMATE2005']]

cross = pd.read_csv(f'{ROOT_DIR}/1data/csv/99mfips.csv')
cross = cross[cross["M"].notna()]
cross['MSA'] = cross.apply(lambda x: x['PMSA'] if not np.isnan(x['PMSA']) else x['MSA'], axis=1)
cross['STATE'] = cross['COUNTY'].apply(lambda x : str(x)[:2] if len(str(x))==5 else ( str(x)[:1] if len(str(x))==4 else 0))
cross = cross[['MSA','STATE','COUNTY','M','PLACE','NAME']]
cross_pt = cross[cross["PLACE"].notna()].reset_index(drop=True)
cross_full = cross[cross["PLACE"].isna()][['MSA','COUNTY','M','NAME']].reset_index(drop=True)
cross_pt['PLACE'] = cross_pt['PLACE'].astype(int)
cross_pt['STATE'] = cross_pt['STATE'].astype(int)

merged = pd.merge(sub_est, cross_full, how='left', on='COUNTY')

merged_pt = merged[ (merged['M'].isna())][['SUMLEV','COUSUB','STATE','STNAME','COUNTY','POPESTIMATE2005']]
merged_pt = pd.merge( merged_pt, cross_pt, how='left', left_on=['STATE','COUNTY','COUSUB'], right_on = ['STATE','COUNTY','PLACE'])

merged_pt_2 = merged_pt[ (merged_pt['M'].isna()) & (merged_pt['PLACE']!=0) ][['SUMLEV','PLACE','COUSUB','STATE','STNAME','COUNTY','POPESTIMATE2005']]
merged_pt_2 = pd.merge(merged_pt_2, cross_pt, how='left', on = ['STATE','COUNTY','PLACE'] )

merged = pd.concat([merged, merged_pt, merged_pt_2], axis=0)
merged = merged[ merged['M'].notna() ]

merged = merged.sort_values(by=['MSA','COUNTY','COUSUB','PLACE'], ascending=False).drop_duplicates(subset=['PLACE','STATE','STNAME','POPESTIMATE2005','COUNTY','MSA','M','NAME'])

remove_list = []
for mcd in list(set(merged['COUSUB']))[1:]:
    df = merged[merged['COUSUB']==mcd].sort_values(by=['PLACE'])
    if (df.iloc[0]['PLACE'] == 0) and ( len(df)>1 ):
        remove_list.append( mcd )

remove_list_balance = []
remove_list_locs = []

for county in list(set(merged['COUNTY'])):
    df = merged[ merged['COUNTY']==county ]

    df2 = df[ df['PLACE']==99990 ] 
    if len(df2)>1 :
        remove_list_balance.append( county )
    
    place_list = list(set( df['PLACE'] ))
    try:
        place_list.remove(0)
    except:
        pass
    try:
        place_list.remove(99990)
    except:
        pass

    if not isinstance(place_list, type(None)):
        for place in place_list:
            df1 = df[ df['PLACE']== place ]
            if (len(df1)>1) and (len(df1[ df1['COUSUB']==0 ])>0):
                remove_list_locs = remove_list_locs + df1.index[ df1['COUSUB']==0 ].tolist()

merged = merged[ ~( (merged['COUSUB'].isin(remove_list)) & (merged['PLACE']==0)) ]
merged = merged[ ~( (merged['COUNTY'].isin(remove_list_balance)) & (merged['PLACE']==99990) & (merged['COUSUB']==0)) ]
merged = merged.loc[~merged.index.isin(remove_list_locs)]

merged.to_csv(f'{ROOT_DIR}3outputs/2005_metro_places.csv', index=False)

######## Build Metro HHI
places = merged

metros = pd.DataFrame()
for metro in list(set(places['MSA'])):
    df = places[ (places['MSA']==metro) & (places['M']==1) ][['PLACE','POPESTIMATE2005']]
    df_balance = df[ (df['PLACE']==99990) | (df['PLACE']==0)]
    df = df[~((df['PLACE']==99990) | (df['PLACE']==0))]
    df = df.groupby(by=['PLACE']).sum()
    df = pd.concat([df,df_balance], axis=0)

    ua_sum_pop = df['POPESTIMATE2005'].sum()
    hhi = 0     
    for index, row in df.iterrows():
        hhi += ((100*row['POPESTIMATE2005']/ua_sum_pop)**2)/100
    
    metros = metros.append({'hhi': hhi , 'pop_est':ua_sum_pop,'msa': metro }, ignore_index=True)

metros = metros.set_index(keys=['msa'])

######## Build weighted WRLURI 
wlr = pd.read_csv(f'{ROOT_DIR}/1data/csv/wlr.csv')
wlr = wlr[ (wlr['msa99']!=9999) & (wlr['WRLURI'].notna())]
wlr = wlr[['name','ufips','state','msaname99','msa99','WRLURI','weight_metro']]

merged = pd.merge(wlr, sub_est[ sub_est['SUMLEV']==162 ][['PLACE','STNAME','POPESTIMATE2005']], how='left', left_on=['ufips','state'], right_on=['PLACE','STNAME'])

merged_1 = merged[ merged['POPESTIMATE2005'].isna() ][['name','ufips','state','msaname99','msa99','WRLURI','weight_metro']]
merged_1 = pd.merge(merged_1, sub_est[sub_est['SUMLEV']==61][['COUSUB','STNAME','POPESTIMATE2005']], how='left', left_on=['ufips','state'], right_on=['COUSUB','STNAME'])
merged = pd.concat([merged, merged_1], axis=0).drop_duplicates(subset=['ufips','state']).drop(columns=['COUSUB'])
merged = merged[ merged['POPESTIMATE2005'].notna() ]

for msa in list(set(merged['msa99'])):
    df = merged[merged['msa99']==msa]
    sum_pop = 0
    sum_weight = 0 
    for index, row in df.iterrows():
        sum_weight += row['weight_metro']*row['POPESTIMATE2005']
        sum_pop += row['POPESTIMATE2005']
    
    df['new_weight'] = df.apply(lambda x: (((x['weight_metro']*x['POPESTIMATE2005'])/sum_weight) * x['WRLURI']) , axis=1)
    df['pop_weight'] = df.apply(lambda x: ((x['POPESTIMATE2005']/sum_pop) * x['WRLURI']) , axis=1)
    
    metros.loc[msa, 'weighted_wrluri'] = df['new_weight'].sum()
    metros.loc[msa, 'pop_wrluri'] = df['pop_weight'].sum()
    metros.loc[msa, 'msaname'] = df.iloc[0]['msaname99'] 

hhi_analysis = metros[metros['weighted_wrluri'].notna() & metros['hhi'].notna()]
hhi_analysis.to_csv(f'{ROOT_DIR}3outputs/metros_wrluri_hhi.csv')

plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('HHI')
ax.set_ylabel('WRLURI')
ax.grid(True)
ax.scatter(hhi_analysis['hhi'], hhi_analysis['weighted_wrluri'], color=[0.0, 0.3, 0])
plt.savefig(f'{ROOT_DIR}/3outputs/hhi_wrluri.png')

# # Saiz
# saiz = pd.read_csv(f'{ROOT_DIR}/1data/csv/saiz2010.csv')
# saiz = saiz[['msa','undep_land','WRI_SAIZ']]

# # Merge
# final = pd.merge(cz, metros, how='left', on='msa')
# final = pd.merge(final, saiz, how='left', on='msa')

# final.to_csv(f'{ROOT_DIR}/3outputs/reg_table.csv', index=False)

# plt.rc('font', size=12)
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_xlabel('weighted_wrluri')
# ax.set_ylabel('undep_land')
# ax.grid(True)

# ax.scatter(final['WRI_SAIZ'], final['undep_land'], color=[0.0, 0.3, 0])

# # ax.legend([f'Parental Income in 75th PCTL',f'Parental Income in 25th PCTL'])

# plt.savefig(f'{ROOT_DIR}/3outputs/task1.png')

##### Run Regression
stata.pdataframe_to_data(hhi_analysis, force=True)

stata.run('''
  est clear
  
  gen hhi2 = hhi*hhi
  
  reg weighted_wrluri hhi hhi2

  * ivreg kfr_pooled_pooled_p25 (weighted_wrluri = undep_land) poor_share2000 hhinc_mean2000 share_black2000 share_white2000 share_hisp2000 share_asian2000 rent_twobed2015 singleparent_share2000 popdensity2000, first
  
  ''')