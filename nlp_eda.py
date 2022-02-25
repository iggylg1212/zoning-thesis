from global_var import *
from config import *
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pymysql
import math

def create_connection():
    conn = None
    errors = 0
    while conn is None and errors<10: 
      try:
          conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
      except:
          errors += 1

    return conn

print('PEARSON CORRELATION INDICES: FULL SAMPLE')
conn = create_connection()
with conn:
  cur = conn.cursor()
  cur.execute('SELECT wrluri_impute, alpha, beta, tau FROM zoning_attributes')
  results = cur.fetchall()
  wrluri = []
  alpha = []
  beta = []
  tau = []
  for result in results:
    wrluri.append(result[0])
    alpha.append(result[1])
    beta.append(result[2])
    tau.append(result[3])
df = pd.DataFrame({'wrluri':wrluri,'alpha':alpha,'beta':beta,'tau':tau})
print(df.corr())

print('PEARSON CORRELATION INDICES: 2018, URBANIZED')
conn = create_connection()
with conn:
  cur = conn.cursor()
  cur.execute('SELECT wrluri_impute, alpha, beta, tau FROM zoning_attributes WHERE urbanized2018=1 AND year=2018')
  results = cur.fetchall()
  wrluri = []
  alpha = []
  beta = []
  tau = []
  for result in results:
    wrluri.append(result[0])
    alpha.append(result[1])
    beta.append(result[2])
    tau.append(result[3])
df = pd.DataFrame({'wrluri':wrluri,'alpha':alpha,'beta':beta,'tau':tau})
print(df.corr())

conn = create_connection()
with conn:
  cur = conn.cursor()
  cur.execute('SELECT place, state, govtype, year, month, day, gisjoin, urbanized1990, urbanized2000, urbanized2010, urbanized2018, wrluri_impute, alpha, beta, tau FROM zoning_attributes')
  results = cur.fetchall()

  place = []
  state = []
  govtype = []
  year = []
  month = []
  day = []
  gisjoin = []
  urbanized1990 = []
  urbanized2000 = []
  urbanized2010 = []
  urbanized2018 = []
  wrluri = []
  alpha = []
  beta = []
  tau = []
  for result in results:
    place.append(result[0])
    state.append(result[1])
    govtype.append(result[2])
    year.append(result[3])
    month.append(result[4])
    day.append(result[5])
    gisjoin.append(result[6])
    urbanized1990.append(result[7])
    urbanized2000.append(result[8])
    urbanized2010.append(result[9])
    urbanized2018.append(result[10])
    wrluri.append(result[11])
    alpha.append(result[12])
    beta.append(result[13])
    tau.append(result[14])
df = pd.DataFrame({'place':place, 'state':state, 'govtype':govtype, 'year':year, 'month':month, 'day':day, 'gisjoin':gisjoin, 'urbanized1990':urbanized1990, 'urbanized2000':urbanized2000, 'urbanized2010':urbanized2010, 'urbanized2018':urbanized2018, 'wrluri':wrluri, 'alpha':alpha, 'beta':beta, 'tau':tau})
df.to_csv(f'{S3_PATH}nlp/csv/nlp_final.csv', index=False)
df = df[(df['urbanized1990']==1) & (df['urbanized2000']==1) & (df['urbanized2010']==1) & (df['urbanized2018']==1)][['place','state','govtype','year','month','day','gisjoin','wrluri','alpha','beta','tau']]
df.to_csv(f'{S3_PATH}nlp/csv/urban1990sample.csv', index=False)
print('SUMMARY STATS')
print(df.describe())

auto_corrs = pd.DataFrame()
for interval in list(range(1,13)):
  year_pair = []
  for year in range(1989,2023-interval):
    year_pair.append([year, year+interval])
  auto = pd.DataFrame()
  gisjoin_set = set(df['gisjoin'])
  for gisjoin in gisjoin_set:
    for years in year_pair:
      muni = df[df['gisjoin']==gisjoin]
      first = muni[muni['year']==years[0]]
      if len(first)!=0:
        first = pd.DataFrame(first[['wrluri','alpha','beta','tau']]).rename(columns={'wrluri':'wrluri_first','alpha':'alpha_first','beta':'beta_first','tau':'tau_first'}).mean()
        last = muni[muni['year']==years[1]]
        if len(last)!=0:
          last = pd.DataFrame(last[['wrluri','alpha','beta','tau']]).rename(columns={'wrluri':'wrluri_last','alpha':'alpha_last','beta':'beta_last','tau':'tau_last'}).mean()
          corr = pd.DataFrame(pd.concat([first,last])).transpose()
          auto = pd.concat([auto,corr])
  auto_corr = auto.corr()
  auto_corr = pd.DataFrame({'lag':[interval],'wrluri':[auto_corr['wrluri_first']['wrluri_last']],'alpha':[auto_corr['alpha_first']['alpha_last']],'beta':[auto_corr['beta_first']['beta_last']],'tau':[auto_corr['tau_first']['tau_last']],'n':[len(auto)]})
  auto_corrs = pd.concat([auto_corrs,auto_corr])
print(auto_corrs)
auto_corrs.to_csv(f'{S3_PATH}nlp/csv/auto_corrs.csv', index=False)

df = pd.read_csv(f'{S3_PATH}nlp/csv/urban1990sample.csv')
gisjoins = set(df['gisjoin'])
new = pd.DataFrame()
for gisjoin in gisjoins:
    muni = df[df['gisjoin']==gisjoin]
    meaned = muni[['wrluri','alpha','beta','tau','year']].groupby('year').mean().reset_index()
    meaned['place'] = muni['place'].iloc[0]
    meaned['state'] = muni['state'].iloc[0]
    meaned['gisjoin'] = muni['gisjoin'].iloc[0]
    meaned['govtype'] = muni['govtype'].iloc[0]
    new = pd.concat([new,meaned])
new.to_csv(f'{S3_PATH}nlp/csv/urban1990meaned.csv', index=False)

# Year Histogram
df = pd.read_csv(f'{S3_PATH}nlp/csv/urban1990meaned.csv')
hist = pd.DataFrame()
gisjoins = set(df['gisjoin'])
for gisjoin in gisjoins:
    muni = df[df['gisjoin']==gisjoin]
    hist = pd.concat([pd.DataFrame({'gisjoin':[muni['gisjoin'].iloc[0]],'len':[len(muni)]}),hist])
for int in list(range(1,1+hist['len'].max())):
    sub = 100*len(hist[hist['len']==int])/len(hist)
    rounded = round(sub,3)
    print(f'{int} years: {rounded}%')
for year in list(range(df['year'].min(),df['year'].max()+1)):
    year_df = 100*len(set(df[df['year']==year]['gisjoin']))/len(set(df['gisjoin']))
    rounded = round(year_df,3)
    print(f'{year}: {rounded}%')

######### ADD COVS for 2018
def import_shpfile(path):
    shp_files = ['.dbf','.prj','.sbn','.sbx','.shp','.shp.xml','.shx']
    for file in shp_files:
        try:
            S3FS.get(f'{S3_PATH}{path}{file}', f'temporary{file}')
        except:
            pass
    df = gpd.read_file('temporary.shp', crs='EPSG:4326')
    for file in shp_files:
        try:
            os.remove(f'temporary{file}')
        except:
            pass
    return df

county = import_shpfile(f'nlp/gis/county_2018/US_county_2018')
places = import_shpfile(f'nlp/gis/place_2018/US_place_2018')
areas = pd.concat([county,places])[['GISJOIN','Shape_Area']]

S3FS.get(f'{S3_PATH}nlp/gis/urb_gov_2018.gpkg',f'urb_gov2018.gpkg')
df = gpd.read_file(f'urb_gov2018.gpkg')
os.remove(f'urb_gov2018.gpkg')
gisjoin_set = list(df['GISJOIN'])

place_covs = pd.read_csv(f'{S3_PATH}nlp/csv/place_covs_2018/nhgis0028_ds239_20185_place.csv', encoding = "ISO-8859-1").rename(columns={'PLACE':'NAME'})
place_covs = place_covs[place_covs['GISJOIN'].isin(gisjoin_set)]
place_covs['source'] = 'place'

county_covs = pd.read_csv(f'{S3_PATH}nlp/csv/county_covs_2018/nhgis0029_ds239_20185_county.csv', encoding = "ISO-8859-1").rename(columns={'COUNTY':'NAME'})
county_covs = county_covs[county_covs['GISJOIN'].isin(gisjoin_set)]
county_covs['source'] = 'county'

covs = pd.concat([place_covs,county_covs]).fillna(0)
covs = covs.merge(areas, how='left', on='GISJOIN')
covs = covs[(covs['AJWBE001']>0) & (covs['AJXKE001']>0) & (covs['AJ1CE001']>0) & (covs['AJZAE001']>0)] # Total population not zero or null

sample = pd.read_csv(f'{S3_PATH}nlp/csv/nlp_final.csv')
sample = sample[(sample['urbanized2018']==1) & (sample['year']==2018)]
gisjoins = set(sample['gisjoin'])
new = pd.DataFrame()
for gisjoin in gisjoins:
    muni = sample[sample['gisjoin']==gisjoin]
    meaned = muni[['wrluri','alpha','beta','tau','year']].groupby('year').mean().reset_index()
    meaned['gisjoin'] = muni['gisjoin'].iloc[0]
    new = pd.concat([new, meaned])
sample = new[['gisjoin','wrluri','alpha','beta','tau']].rename(columns={'gisjoin':'GISJOIN'})

cov_list = ['GISJOIN','NAME','STATE','source','log_total_pop','log_density','log_median_income','female_perc','over_65_perc','hispan_perc','white_perc','black_perc','asian_perc','families_perc','some_college_perc','bachelors_perc','masters_perc','poverty_perc','unemployed_perc','employed_perc','homeowner_perc']
def proc_data(df):
    df['log_total_pop'] = df['AJWBE001'].apply(lambda x: math.log(x))
    df['log_density'] = df.apply(lambda x: math.log(x['AJWBE001']/x['Shape_Area']), axis=1)
    df['log_median_income'] = df['AJZAE001'].apply(lambda x: math.log(x))
    df['female_perc'] = df.apply(lambda x: x['AJWBE026']/x['AJWBE001'], axis=1)
    df['over_65_perc'] = df.apply(lambda x: (x['AJWBE020']+x['AJWBE021']+x['AJWBE022']+x['AJWBE023']+x['AJWBE024']+x['AJWBE025']+x['AJWBE044']+x['AJWBE045']+x['AJWBE046']+x['AJWBE047']+x['AJWBE048']+x['AJWBE049'])/x['AJWBE001'], axis=1)
    df['hispan_perc'] = df.apply(lambda x: x['AJWVE012']/x['AJWVE001'], axis=1)
    df['white_perc'] = df.apply(lambda x: x['AJWVE003']/x['AJWVE001'], axis=1)
    df['black_perc'] = df.apply(lambda x: x['AJWVE004']/x['AJWVE001'], axis=1)
    df['asian_perc'] = df.apply(lambda x: x['AJWVE006']/x['AJWVE001'], axis=1)
    df['families_perc'] = df.apply(lambda x: x['AJXKE002']/x['AJXKE001'], axis=1)
    df['some_college_perc'] = df.apply(lambda x: (x['AJYPE019']+x['AJYPE020']+x['AJYPE021'])/x['AJYPE001'], axis=1)
    df['bachelors_perc'] = df.apply(lambda x: x['AJYPE022']/x['AJYPE001'], axis=1)
    df['masters_perc'] = df.apply(lambda x: (x['AJYPE025']+x['AJYPE024']+x['AJYPE023'])/x['AJYPE001'], axis=1)
    df['poverty_perc'] = df.apply(lambda x: x['AJY4E001']/x['AJWBE001'], axis=1)
    df['unemployed_perc'] = df.apply(lambda x: x['AJ1CE005']/x['AJ1CE001'], axis=1)
    df['employed_perc'] = df.apply(lambda x: x['AJ1CE004']/x['AJ1CE001'], axis=1)
    df['homeowner_perc'] = df.apply(lambda x: x['AJ1UE002']/x['AJ1UE001'], axis=1)
    return df

covs = proc_data(covs)[cov_list]
df = covs.merge(sample, how='left', on='GISJOIN')
df['sample'] = df.apply(lambda x: 0 if math.isnan(x['wrluri']) else 1, axis=1)
df.to_csv(f'{S3_PATH}nlp/csv/2018_urb_gov_covs.csv',index=False)

######### RUN HECKMAN REGRESSIONS 2018
df = pd.read_csv(f'{S3_PATH}nlp/csv/2018_urb_gov_covs.csv')
df['place'] = df.apply(lambda x: 1 if x['source']=='place' else 0, axis=1)

import stata_setup
stata_setup.config('/Applications/Stata/', 'be')
from pystata import stata

stata.pdataframe_to_data(df, force=True)
stata.run('''
est clear

lab var wrluri "WRLURI"
lab var alpha "Alpha"
lab var beta "Beta"
lab var tau "Tau"
lab var sample "In Sample"

lab var log_total_pop "Ln Total Pop."
lab var place "Municipality Flag"
lab var log_density "Ln Density (Ind/Sq Mi)"
lab var log_median_income "Ln Median Income"
lab var female_perc "Female (\%)"
lab var over_65_perc "Over 65 y.o. (\%)"
lab var hispan_perc "Hispanic (\%)"
lab var white_perc "White (\%)"
lab var black_perc "Black (\%)"
lab var asian_perc "Asian (\%)"
lab var families_perc "Family HH (\%)"
lab var some_college_perc "Some College (\%)"
lab var bachelors_perc "Bachelors Deg. (\%)"
lab var masters_perc "Masters Deg. (\%)"
lab var poverty_perc "Pop. in Poverty (\%)"
lab var unemployed_perc "Unemployed (\% W.A.)"
lab var employed_perc "Employed (\% W.A.)"
lab var homeowner_perc "Homeowners (\% HH)"

local list " log_total_pop log_density log_median_income place female_perc over_65_perc hispan_perc white_perc black_perc asian_perc families_perc some_college_perc bachelors_perc masters_perc poverty_perc unemployed_perc employed_perc homeowner_perc "

eststo: heckman wrluri `list', select(sample = `list')
eststo: heckman alpha `list', select(sample = `list')
eststo: heckman beta `list', select(sample = `list')
eststo: heckman tau `list', select(sample = `list')

esttab using "heckman.tex", replace ///
b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
title("Regulatory Indices with Heckman Correction (2018)") ///
label long booktabs alignment(D{.}{.}{-1}) collabels(none)
''')
S3FS.put('heckman.tex',f'{S3_PATH}outputs/tables/heckman.tex')
os.remove('heckman.tex')

## Top/Bottom 10
df = pd.read_csv(f'{S3_PATH}nlp/csv/2018_urb_gov_covs.csv')
places = df[(df['sample']==1) & (df['source']=='place')]
county = df[(df['sample']==1) & (df['source']=='county')]

places.sort_values(by='wrluri', ascending=False, inplace=True)
county.sort_values(by='wrluri', ascending=False, inplace=True)
print(places.iloc[:10][['NAME','STATE','wrluri']])
print(places.iloc[-10:][['NAME','STATE','wrluri']])
print(county.iloc[:10][['NAME','STATE','wrluri']])
print(county.iloc[-10:][['NAME','STATE','wrluri']])

places.sort_values(by='alpha', ascending=False, inplace=True)
county.sort_values(by='alpha', ascending=False, inplace=True)
print(places.iloc[:10][['NAME','STATE','alpha']])
print(places.iloc[-10:][['NAME','STATE','alpha']])
print(county.iloc[:10][['NAME','STATE','alpha']])
print(county.iloc[-10:][['NAME','STATE','alpha']])

places.sort_values(by='beta', ascending=False, inplace=True)
county.sort_values(by='beta', ascending=False, inplace=True)
print(places.iloc[:10][['NAME','STATE','beta']])
print(places.iloc[-10:][['NAME','STATE','beta']])
print(county.iloc[:10][['NAME','STATE','beta']])
print(county.iloc[-10:][['NAME','STATE','beta']])

places.sort_values(by='tau', ascending=False, inplace=True)
county.sort_values(by='tau', ascending=False, inplace=True)
print(places.iloc[:10][['NAME','STATE','tau']])
print(places.iloc[-10:][['NAME','STATE','tau']])
print(county.iloc[:10][['NAME','STATE','tau']])
print(county.iloc[-10:][['NAME','STATE','tau']])

## Time Series
df = pd.read_csv(f'{S3_PATH}nlp/csv/nlp_final.csv')
df = df[(df['urbanized2010']==1) & (df['urbanized2018']==1)]
gisjoins = set(df['gisjoin'])
new = pd.DataFrame()
for gisjoin in gisjoins:
    muni = df[df['gisjoin']==gisjoin]
    meaned = muni[['wrluri','alpha','beta','tau','year']].groupby('year').mean().reset_index()
    meaned['gisjoin'] = muni['gisjoin'].iloc[0]
    meaned['govtype'] = muni['govtype'].iloc[0]
    new = pd.concat([new,meaned])
sample = new[['gisjoin','govtype','year','wrluri','alpha','beta','tau']]

place_joins = set(sample[sample['govtype']=='place']['gisjoin'])
keep = []
for join in place_joins:
    muni = sample[sample['gisjoin']==join]
    if (muni['year'].iloc[0]<=2010) and (len(muni)>=10):
        keep.append(muni['gisjoin'].iloc[0])
place_series = sample[(sample['gisjoin'].isin(keep)) & (sample['year']>=2010)].groupby('year').mean().reset_index()
print(place_series)

plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_title('Places, Mean Index Values 2010-2021')
ax.grid(True)
ax.plot(place_series['year'], place_series['wrluri'], color='tab:blue')
ax.plot(place_series['year'], place_series['alpha'], color='tab:red')
ax.plot(place_series['year'], place_series['beta'], color='tab:green')
ax.plot(place_series['year'], place_series['tau'], color='tab:orange')
ax.legend(['WRLURI','Alpha','Beta','Tau'])
plt.savefig('mean_places_2010-2021.png')

county_joins = set(sample[sample['govtype']=='county']['gisjoin'])
keep = []
for join in county_joins:
    muni = sample[sample['gisjoin']==join]
    if (muni['year'].iloc[0]<=2010) and (len(muni)>=10):
        keep.append(muni['gisjoin'].iloc[0])
county_series = sample[(sample['gisjoin'].isin(keep)) & (sample['year']>=2010)].groupby('year').mean().reset_index()
print(county_series)

plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_title('Counties, Mean Index Values 2010-2021')
ax.grid(True)
ax.plot(county_series['year'], county_series['wrluri'], color='tab:blue')
ax.plot(county_series['year'], county_series['alpha'], color='tab:red')
ax.plot(county_series['year'], county_series['beta'], color='tab:green')
ax.plot(county_series['year'], county_series['tau'], color='tab:orange')
ax.legend(['WRLURI','Alpha','Beta','Tau'])
plt.savefig('mean_county_2010-2021.png')

S3FS.put('mean_places_2010-2021.png',f'{S3_PATH}outputs/tables/mean_places_2010-2021.png')
S3FS.put('mean_county_2010-2021.png',f'{S3_PATH}outputs/tables/mean_county_2010-2021.png')
os.remove('mean_places_2010-2021.png')
os.remove('mean_county_2010-2021.png')