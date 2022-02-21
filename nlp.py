from global_var import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.constraints import NonNeg
import tensorflow as tf
from contextlib import closing
import scipy
from config import *
from joblib import dump, load
import math
import csv
import tqdm
import pymysql
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, Array
import re
import random
from sentence_transformers import SentenceTransformer, losses, InputExample, util, evaluation
from torch.utils.data import DataLoader

def create_connection():
    conn = None
    errors = 0
    while conn is None and errors<10: 
      try:
          conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
      except:
          errors += 1

    return conn

def create_row(conn, row):
    cur = conn.cursor()
    cur.execute(''' CREATE TABLE IF NOT EXISTS zoning (
                      place TEXT NOT NULL,
                      state TEXT NOT NULL,
                      year INTEGER NOT NULL, 
                      month INTEGER NOT NULL, 
                      day INTEGER NOT NULL, 
                      zone_text LONGTEXT NOT NULL,
                      gisjoin TEXT,
                      govtype TEXT,
                      urbanized1990 BOOL DEFAULT 0,
                      urbanized2000 BOOL DEFAULT 0,
                      urbanized2010 BOOL DEFAULT 0, 
                      urbanized2018 BOOL DEFAULT 0   
                    ); ''')
    conn.commit()
    cur.execute("INSERT INTO zoning (place, state, year, month, day, zone_text) values (%s,%s,%s,%s,%s,%s);", row)
    conn.commit()

conn = create_connection()
with conn:
  cur = conn.cursor()
  cur.execute('SELECT place, state, year, month, day FROM zoning')
  done_list = set(cur.fetchall())

zoning_phrases = ['development', 'development code', 'zoning', 'architectural', 'infill', 'density', 'housing', 'land', 'growth areas', 'building', 'preservation', 'historic', 'land use', 'neighborhood', 'subdivision', 'planning', 'property maintenance']
directory = S3FS.ls(f'{S3_PATH}nlp/csv/municode')
tasks = directory

model_name = 'paraphrase-MiniLM-L3-v2'
model_save_path = f'continue_training-{model_name}'
train_batch_size = 16
num_epochs = 4

def create_train(df):
  train_samples = []
  dev_samples = []
  for idx, row in df.iterrows():
    for column in df.columns.to_list()[1:]:
      inp_example = InputExample(texts=[row['Text'], column], label=float(row[column]))
      if random.randrange(0, 10) < 2:
        dev_samples.append(inp_example)
      else:
        train_samples.append(inp_example)
  return train_samples, dev_samples

fine_tuning = pd.read_csv(f'{S3_PATH}nlp/csv/fine_tuning_semantic.csv')
train_samples, dev_samples = create_train(fine_tuning)

model = SentenceTransformer(model_name)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=num_epochs, evaluation_steps=1000, warmup_steps=warmup_steps, output_path=model_save_path)

model = SentenceTransformer(model_save_path)
test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)

def semantic_similarity(embed, sen):
    sen = model.encode(sen)
    cosine_scores = util.pytorch_cos_sim(embed, sen)
    return float(cosine_scores[0][0])

CLEANR = re.compile('<.*?>|\n|d-|____________|_____|\t|---')
def cleaner(raw_text):
  try:
    cleantext = ' '.join(re.sub(CLEANR, '', raw_text).split())
    cleantext = re.sub("[^A-Za-z0-9 .,!?:;']", ' ', cleantext).replace('Content is too large for cell.', '').replace('  ',' ').replace('  ',' ')
  except:
    cleantext = raw_text  
  return cleantext

def text_creator(df):
  try:
    df['raw_text'] = df.apply(lambda x: f'{x["Title"]} {x["Subtitle"]} {x["Content"]} ', axis=1)
    text = df['raw_text'].str.cat(sep=' ')
  except:
    text = ''
  return text

def zoning_creator(df, indices):
    zoning = pd.DataFrame(columns=['Title','Subtitle','Content'])
    zoning_idx = []
    for i in range(len(indices)):
      index = indices[i]
      if df.iloc[index]['Subtitle'] != '':
        encoded = df.iloc[index]['Subtitle'].lower()
      else:
        encoded = df.iloc[index]['Content'].lower()
      phrase_list = [encoded]
      encode = model.encode(encoded)
      high_score = 0
      for phrase in zoning_phrases:
        score = semantic_similarity(encode, phrase)
        phrase_list.append(score)
        if score >= high_score:
          high_score = score
      if high_score < .9:
        phrase_list = []    
      else:
        if (i+1) == len(indices):
          zoning_idx = zoning_idx+list(range(index,len(df.index)))
          new = df.iloc[index:]
        else:
          zoning_idx = zoning_idx+list(range(index,indices[i+1]))
          new = df.iloc[index:indices[i+1]]
        
        zoning = zoning.append(new)
      
    return zoning, zoning_idx

def loop(filename):
    ## Attributes
    block_title = re.findall('[A-Z][^A-Z]*', filename.replace('Compilation-','').split('/')[-1])
    date = re.sub("[^0-9]", "", block_title[-1])
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    state = ''.join(block_title[-10:-8])
    muni = ' '.join(block_title[:-10])
    
    tup = (muni, state, year, month, day)
    if tup in done_list:
      return

    ## Text Data
    df = pd.read_excel(f's3://{filename}')
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(columns = ['NodeId','Url'])
    df = df.applymap(lambda x: cleaner(x)).fillna('')
    
    df = df[~((df['Title'].str.lower()).str.contains('table'))].reset_index(drop=True)

    zone_text = pd.DataFrame(columns=['Title','Subtitle','Content'])

    app_idx = df[(df['Title'].str.lower()).str.contains('appendi')].index.tolist()

    if len(app_idx)>=1:
      body = df.iloc[:app_idx[0]].reset_index(drop=True)
    else:
      body = df.reset_index(drop=True)
    
    structure = ['title', 'chapter', 'article', 'division', 'subdivision']
    remainder = body
    for struct in structure:
      indices = remainder[(remainder['Title'].str.lower()).str.contains(struct)].index.tolist()
      if len(indices)==0:
        continue
      zoning, zoning_idx = zoning_creator(remainder, indices)
      remainder = remainder.loc[~remainder.index.isin(zoning_idx)].reset_index(drop=True)
      zone_text = zone_text.append(zoning)
    
    indices = remainder.index.tolist()
    zoning, zoning_idx = zoning_creator(remainder, indices)
    remainder = remainder.loc[~remainder.index.isin(zoning_idx)]
    zone_text = zone_text.append(zoning)

    if len(app_idx)>=1:
      zoning, zoning_idx = zoning_creator(df, app_idx)
      zone_text = zone_text.append(zoning)

    zone_text = text_creator(zone_text)

    conn = create_connection()
    with conn:
      row = [muni, state, year, month, day, zone_text]
      create_row(conn, row)

if __name__ == "__main__":
  pool = Pool()
  for _ in tqdm.tqdm(pool.imap_unordered(loop, tasks), total=len(tasks)):
    pass

S3FS.put('continue_training-paraphrase-MiniLM-L3-v2',f'{S3_PATH}nlp/models/corpus_select', recursive=True)
os.remove('continue_training-paraphrase-MiniLM-L3-v2')

# Adding GISJOINs to zoning_corpus
us_state_to_abbrev = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC", "American Samoa": "AS", "Guam": "GU", "Northern Mariana Islands": "MP", "Puerto Rico": "PR", "United States Minor Outlying Islands": "UM", "U.S. Virgin Islands": "VI"}

county_code_1990 = pd.read_csv(f'{S3_PATH}nlp/csv/1990_population/nhgis0017_ds120_1990_county.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','ANPSADPI']].rename(columns={'ANPSADPI':'COUNTY'})
place_code_1990 = pd.read_csv(f'{S3_PATH}nlp/csv/1990_population/nhgis0017_ds120_1990_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','PLACE']]
county_code_2000 = pd.read_csv(f'{S3_PATH}nlp/csv/2000_population/nhgis0023_ds146_2000_county.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','NAME']].rename(columns={'NAME':'COUNTY'})
place_code_2000 = pd.read_csv(f'{S3_PATH}nlp/csv/2000_population/nhgis0023_ds146_2000_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','PLACE']]
county_code_2010 = pd.read_csv(f'{S3_PATH}nlp/csv/2010_population/nhgis0019_ds172_2010_county.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','NAME']].rename(columns={'NAME':'COUNTY'})
place_code_2010 = pd.read_csv(f'{S3_PATH}nlp/csv/2010_population/nhgis0019_ds172_2010_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','PLACE']]
county_code_2018 = pd.read_csv(f'{S3_PATH}nlp/csv/2018_population/nhgis0022_ds239_20185_county.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','COUNTY']]
place_code_2018 = pd.read_csv(f'{S3_PATH}nlp/csv/2018_population/nhgis0021_ds239_20185_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','PLACE']]

codes = pd.concat([county_code_1990, county_code_2000, county_code_2010, county_code_2018, place_code_1990, place_code_2000, place_code_2010, place_code_2018], axis=0)
codes['STATE'] = codes['STATE'].replace(us_state_to_abbrev)
codes['PLACE'] = codes['PLACE'].fillna('').apply(lambda x: ' '.join(x.split(' ')[:-1]))
codes['COUNTY'] = codes['COUNTY'].fillna('')
codes = codes.drop_duplicates()

place_list = str(set(codes[codes['PLACE']!='']['GISJOIN'])).replace('}',')').replace('{','(')
county_list = str(set(codes[codes['COUNTY']!='']['GISJOIN'])).replace('}',')').replace('{','(')

conn = create_connection()
with conn:
    cur = conn.cursor()
    cur.execute('ALTER TABLE zoning ADD uuid TEXT')
    conn.commit()
    cur.execute("UPDATE zoning SET uuid=UUID();")
    conn.commit()
    cur.execute('CREATE TABLE IF NOT EXISTS zoning_attributes SELECT place, state, year, month, day, gisjoin, govtype, urbanized1990, urbanized2000, urbanized2010, urbanized2018, uuid FROM zoning')
    conn.commit()
    for id, row in codes[codes['COUNTY']!=''].iterrows():
        county = row['COUNTY'].replace("'","''")
        cur.execute(f"UPDATE zoning_attributes SET gisjoin='{row['GISJOIN']}' WHERE place='{county}' AND state='{row['STATE']}'")
        conn.commit()
    for id, row in codes[codes['PLACE']!=''].iterrows():
      place = row["PLACE"].replace("'","''")
      cur.execute(f"UPDATE zoning_attributes SET gisjoin = '{row['GISJOIN']}' WHERE place ='{place}' AND state='{row['STATE']}'")
      conn.commit()
    for id, row in codes[codes['PLACE']!=''].iterrows():
      place = row["PLACE"].replace("'","''")
      cur.execute(f"UPDATE zoning_attributes SET gisjoin='{row['GISJOIN']}' WHERE gisjoin is NULL and place LIKE '%{place}%' AND state='{row['STATE']}'")
      conn.commit()
    for id, row in codes[(codes['COUNTY']!='') & (codes['STATE']=='LA')].iterrows(): # Correcting 'police jury'
      county = row['COUNTY'].replace("'","''")
      cur.execute(f"UPDATE zoning_attributes SET gisjoin='{row['GISJOIN']}' WHERE gisjoin is NULL and place LIKE '%{county}%' AND state='LA'")
      conn.commit()
    
    cur.execute('SELECT place, state FROM zoning_attributes WHERE gisjoin IS NULL')
    null_set = set(cur.fetchall())
    dataframe = pd.DataFrame(columns=['og_place','og_state'])
    for set in null_set:
        dataframe = dataframe.append({'og_place':set[0],'og_state':set[1]}, ignore_index=True)
    dataframe.to_csv('broken.csv')

    corrected = pd.read_csv('broken.csv').fillna('')
    for id, row in corrected.iterrows():
        og_place = row['og_place'].replace("'","''")
        new_place = row['new_place'].replace("'","''")
        cur.execute(f"UPDATE zoning_attributes SET place='{new_place}', state='{row['new_state']}' WHERE place='{og_place}' AND state='{row['og_state']}'")
        conn.commit()
    
    len = len(codes[codes['COUNTY']!=''])+len(codes[codes['PLACE']!=''])
    number = 0
    for id, row in codes[codes['COUNTY']!=''].iterrows():
        number += 1
        print(round(number/len, 4))
        county = row['COUNTY'].replace("'","''")
        cur.execute(f"UPDATE zoning_attributes SET gisjoin = '{row['GISJOIN']}' WHERE gisjoin is NULL AND place='{county}' AND state='{row['STATE']}'")
        conn.commit()
    for id, row in codes[codes['PLACE']!=''].iterrows():
        number += 1
        print(round(number/len, 4))
        place = row["PLACE"].replace("'","''")
        cur.execute(f"UPDATE zoning_attributes SET gisjoin = '{row['GISJOIN']}' WHERE gisjoin is NULL AND place ='{place}' AND state='{row['STATE']}'")
        conn.commit()
    
    cur.execute(f"UPDATE zoning_attributes SET govtype = 'county' WHERE gisjoin in {county_list}")
    cur.execute(f"UPDATE zoning_attributes SET govtype = 'place' WHERE gisjoin in {place_list}")
    conn.commit()

# Finding places and counties in urbanized areas
def create_urb_place(year):
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

    urbs = import_shpfile(f'nlp/gis/urb_area_{year}/US_urb_area_{year}').to_crs('EPSG:4326')

    county = import_shpfile(f'nlp/gis/county_{year}/US_county_{year}').to_crs('EPSG:4326')
    county['source'] = 'county'
    places = import_shpfile(f'nlp/gis/place_{year}/US_place_{year}').to_crs('EPSG:4326')
    places['source'] = 'place'
    govs = pd.concat([county, places])[['GISJOIN','source','geometry']]

    urb_gov = gpd.GeoDataFrame(columns=['GISJOIN','source','geometry'], crs='EPSG:4326')
    for row in list(range(len(urbs))):
        print(round(row/len(urbs),4))
        city = urbs.iloc[row][['geometry']]
        govs['bool'] = govs.intersects(city['geometry'])
        set = govs[govs['bool'] == True][['GISJOIN','source','geometry']]
        urb_gov = pd.concat([urb_gov, set])
        print(urb_gov)

    urb_gov.to_file('urb_gov.gpkg')
    S3FS.put('urb_gov.gpkg',f'{S3_PATH}nlp/gis/urb_gov_{year}.gpkg')
    os.remove('urb_gov.gpkg')
## 1990
create_urb_place('1990')
## 2000
create_urb_place('2000')
## 2010
create_urb_place('2010')
## 2018
create_urb_place('2018')

# Find urban entries
def import_urb_gpkg(filepath):
    S3FS.get(f'{S3_PATH}{filepath}', 'urb_gov.gpkg')
    urb_gov = gpd.read_file('urb_gov.gpkg').to_crs('EPSG:4326')
    os.remove('urb_gov.gpkg')
    urb_joins = str(set(urb_gov['GISJOIN'])).replace('{','').replace('}','')
    return urb_joins

urb_joins1990 = import_urb_gpkg('nlp/gis/urb_gov_1990.gpkg')
urb_joins2000 = import_urb_gpkg('nlp/gis/urb_gov_2000.gpkg')
urb_joins2010 = import_urb_gpkg('nlp/gis/urb_gov_2010.gpkg')
urb_joins2018 = import_urb_gpkg('nlp/gis/urb_gov_2018.gpkg')

conn = create_connection()
with conn:
    cur = conn.cursor()
    cur.execute(f"UPDATE zoning_attributes SET urbanized1990=1 WHERE gisjoin IN ({urb_joins1990})")
    conn.commit()
    cur.execute(f"UPDATE zoning_attributes SET urbanized2000=1 WHERE gisjoin IN ({urb_joins2000})")
    conn.commit()
    cur.execute(f"UPDATE zoning_attributes SET urbanized2010=1 WHERE gisjoin IN ({urb_joins2010})")
    conn.commit()
    cur.execute(f"UPDATE zoning_attributes SET urbanized2018=1 WHERE gisjoin IN ({urb_joins2018})")
    conn.commit()

############################### Learn WRLURI
# Adding WRLURI Data to DB
place_code_1990 = pd.read_csv(f'{S3_PATH}nlp/csv/1990_population/nhgis0017_ds120_1990_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATEA','PLACEA']]
place_code_2000 = pd.read_csv(f'{S3_PATH}nlp/csv/2000_population/nhgis0023_ds146_2000_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATEA','PLACEA']]
place_code_2010 = pd.read_csv(f'{S3_PATH}nlp/csv/2010_population/nhgis0019_ds172_2010_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATEA','PLACEA']]
place_code_2018 = pd.read_csv(f'{S3_PATH}nlp/csv/2018_population/nhgis0021_ds239_20185_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATEA','PLACEA']]

codes = pd.concat([place_code_1990, place_code_2000, place_code_2010, place_code_2018], axis=0)
codes = codes.drop_duplicates()

key = pd.read_csv(f'{S3_PATH}nlp/csv/wrluri_2018_key.csv')
var_list = list(key['varname'])
var_list.extend(('WRLURI18','fipsplacecode18','statecode'))

wrluri = pd.read_stata(f'{S3_PATH}nlp/stata/WRLURI_01_15_2020.dta')[var_list].rename({'WRLURI18':'WRLURI'})

for index, row in key.iterrows():
  if row['value']=='1,6,2':
    wrluri[row['varname']] = wrluri[row['varname']].replace({1:6, 6:2, 2:1})
  if row['value']=='1,2':
    wrluri[row['varname']] = wrluri[row['varname']].replace({2:1,1:2})
  if row['value']=='2,1,3':
    wrluri[row['varname']] = wrluri[row['varname']].replace({2:3,1:2,3:1})
  if row['value']=='0,1':
    wrluri[row['varname']] = wrluri[row['varname']].replace({0:1,1:0})

wrluri = wrluri.merge(codes, how = 'left', left_on=['fipsplacecode18','statecode'], right_on=['PLACEA','STATEA'])
wrluri = wrluri[wrluri['GISJOIN'].notna()]
wrluri_join = wrluri.replace(np.nan, 'NULL')

conn = create_connection()
with conn:
    cur = conn.cursor()
    cur.execute('ALTER TABLE zoning_attributes ADD COLUMN WRLURI FLOAT')
    conn.commit()
    number = 0
    for id, row in wrluri.iterrows():
        number += 1
        print(round(number/len(wrluri),4))
        cur.execute(f"UPDATE zoning_attributes SET WRLURI={row['WRLURI']} WHERE gisjoin='{str(row['GISJOIN'])}' AND year=2018")
        conn.commit()

columns___ = list(wrluri.columns)
remove_list = ['WRLURI18', 'fipsplacecode18','statecode', 'GISJOIN', 'STATEA', 'PLACEA']
for remove in remove_list:
  columns___.remove(remove)
wrluri = pd.concat([(wrluri[columns___]/wrluri[columns___].max()), wrluri[remove_list]],axis=1)

a_list = list(key[key['bin']=='a']['varname'])
a_list.remove('q1018')
a_list.remove('q618')
a_com = wrluri[a_list].dropna()
a_corr = a_com.corr()
a_corr.to_csv(f'{S3_PATH}nlp/csv/a_corr.csv')
index_keep = list(a_com.index)
stop = "'"
a_string_columns = f"ALTER TABLE zoning_attributes ADD COLUMN { str(a_list).replace(stop,'').replace('[','').replace(']','').replace(',',', ADD COLUMN').replace(',',' FLOAT,') } FLOAT"
a_list.extend('GISJOIN')
wrluri_a = wrluri.loc[index_keep].replace(np.nan, 'NULL')

conn = create_connection()
with conn:
    cur = conn.cursor()
    # cur.execute(a_string_columns)
    # conn.commit()
    number = 0
    for id, row in wrluri_a.iterrows():
        number += 1
        print(round(number/len(wrluri_a),4))
        cur.execute(f"UPDATE zoning_attributes SET q8a18={row['q8a18']}, q8b18={row['q8b18']}, q8c18={row['q8b18']},  q8d18={row['q8d18']},  q8e18={row['q8e18']},  q8f18={row['q8f18']},  q1918 ={row['q1918']},  q19_rezoning18={row['q19_rezoning18']} WHERE gisjoin='{str(row['GISJOIN'])}' AND year=2018")
        conn.commit()

b_list = list(key[key['bin']=='b']['varname'])
b_list.remove('q718')
b_list.remove('q9b18')
b_com = wrluri[b_list].dropna()
b_corr = b_com.corr()
b_corr.to_csv(f'{S3_PATH}nlp/csv/b_corr.csv')
index_keep = list(b_com.index)
stop = "'"
b_string_columns = f"ALTER TABLE zoning_attributes ADD COLUMN { str(b_list).replace(stop,'').replace('[','').replace(']','').replace(',',', ADD COLUMN').replace(',',' FLOAT,') } FLOAT"
b_list.extend('GISJOIN')
wrluri_b = wrluri.loc[index_keep].replace(np.nan, 'NULL')

conn = create_connection()
with conn:
    cur = conn.cursor()
    # cur.execute(b_string_columns)
    # conn.commit()
    number = 0
    for id, row in wrluri_b.iterrows():
        number += 1
        print(round(number/len(wrluri_b),4))
        cur.execute(f"UPDATE zoning_attributes SET q5b18={row['q5b18']}, q5b_multi18={row['q5b_multi18']}, q7b18={row['q7b18']} WHERE gisjoin='{str(row['GISJOIN'])}' AND year=2018")
        conn.commit()

t_list = list(key[key['bin']=='t']['varname'])
remove_list = ['q16b218','q17a218','q17b218','q4_1j18','q4_2t18','q5a_m5018','q17b118','q2118','q4_2k18','q4_2m18','q9c18','q16a118','q16b118','q17a118','q20a18','q20b18','q20c18','q22a18','q22b18','q22c18','q5a18','q4_2s18','q21_subdivision18']
for remove in remove_list:
  t_list.remove(remove)
t_com = wrluri[t_list].dropna()
t_corr = t_com.corr()
t_corr.to_csv(f'{S3_PATH}nlp/csv/t_corr.csv')
index_keep = list(t_com.index)
stop = "'"
t_string_columns = f"ALTER TABLE zoning_attributes ADD COLUMN { str(t_list).replace(stop,'').replace('[','').replace(']','').replace(',',', ADD COLUMN').replace(',',' FLOAT,') } FLOAT"
t_list.extend('GISJOIN')
wrluri_t = wrluri.loc[index_keep].replace(np.nan, 'NULL')

conn = create_connection()
with conn:
    cur = conn.cursor()
    # cur.execute(t_string_columns)
    # conn.commit()
    number = 0
    for id, row in wrluri_t.iterrows():
        number += 1
        print(round(number/len(wrluri_t),4))
        cur.execute(f"UPDATE zoning_attributes SET q4_1a18={row['q4_1a18']}, q4_1b18={row['q4_1b18']}, q4_1c18={row['q4_1c18']}, q4_1d18={row['q4_1d18']}, q4_1e18={row['q4_1e18']}, q4_1f18={row['q4_1f18']}, q4_1g18={row['q4_1g18']}, q4_1h18={row['q4_1h18']}, q4_1i18={row['q4_1i18']}, q4_2l18 ={row['q4_2l18']}, q4_2n18={row['q4_2n18']}, q4_2o18={row['q4_2o18']}, q4_2p18={row['q4_2p18']}, q4_2q18={row['q4_2q18']}, q4_2r18={row['q4_2r18']}, q9a18={row['q9a18']} WHERE gisjoin='{str(row['GISJOIN'])}' AND year=2018")
        conn.commit()

# TFIDF Vectorization
uuid_list = []
corpus_list = []
reg_list = []
conn = create_connection()
with conn:
    cur = conn.cursor()
    cur.execute('SELECT uuid, WRLURI FROM zoning_attributes')
    results = cur.fetchall()
    for result in results:
        uuid_list.append(result[0])
        reg_list.append(result[1])
    uuid_list_str = str(uuid_list).replace('[','').replace(']','')
    cur.execute(f'SELECT zone_text FROM zoning_corpus WHERE uuid in ({uuid_list_str}) ORDER BY FIELD(uuid, {uuid_list_str})')
    results = cur.fetchall()
    for result in results:
        corpus_list.append(result[0])

tfidf = TfidfVectorizer(strip_accents = None, lowercase = True, analyzer='word', stop_words='english', ngram_range=(1,3)).fit_transform(corpus_list)
scipy.sparse.save_npz('tfidf.npz', tfidf)

# WRLURI NN Training
index_list = []
for index in list(range(len(reg_list))):
  if reg_list[index] is not None:
    index_list.append(index)

X = scipy.sparse.load_npz('tfidf.npz')[index_list]
y = np.asarray(reg_list)[index_list]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

regr = MLPRegressor(random_state=1, hidden_layer_sizes=(20,20)).fit(X_train, y_train)
dump(regr, 'wrluri_model.joblib')
print(regr.score(X_test, y_test))

# Imputing WRLURI
conn = create_connection()
with conn:
    cur = conn.cursor()
    cur.execute('ALTER TABLE zoning_attributes ADD COLUMN wrluri_impute FLOAT')
    conn.commit()

regr = load('wrluri_model.joblib')
X = scipy.sparse.load_npz('tfidf.npz')

def impute_worker(index):
  row = X[index]
  imputation = regr.predict(row)[0]
  conn = create_connection()
  conn.__enter__()
  cur = conn.cursor()
  cur.execute(f"UPDATE zoning_attributes SET wrluri_impute={imputation} WHERE uuid='{uuid_list[index]}'")
  conn.commit()
  conn.__exit__()

tasks = list(range(len(uuid_list)))
for task in tqdm.tqdm(tasks):
  impute_worker(task)

S3FS.put('wrluri_model.joblib',f'{S3_PATH}nlp/models/wrluri_model.joblib')
os.remove('wrluri_model.joblib')

############################### Creating Index
key = pd.read_csv(f'{S3_PATH}nlp/csv/wrluri_2018_key.csv')

a_list = list(key[key['bin']=='a']['varname'])
a_list.remove('q1018')
a_list.remove('q618')

b_list = list(key[key['bin']=='b']['varname'])
b_list.remove('q718')
b_list.remove('q9b18')

t_list = list(key[key['bin']=='t']['varname'])
remove_list = ['q16b218','q17a218','q17b218','q4_1j18','q4_2t18','q5a_m5018','q17b118','q2118','q4_2k18','q4_2m18','q9c18','q16a118','q16b118','q17a118','q20a18','q20b18','q20c18','q22a18','q22b18','q22c18','q5a18','q4_2s18','q21_subdivision18']
for remove in remove_list:
  t_list.remove(remove)

def sparse_converter(X):
  coo = X.tocoo()
  indices = np.mat([coo.row, coo.col]).transpose()
  return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))

def create_index(param):
  if param=='alpha':
    str_list = a_list
  if param=='beta':
    str_list = b_list
  if param=='tau':
    str_list = t_list
  
  stop = "', '"
  stop1 = "'"

  list_index = []
  reg_array = []
  conn = create_connection()
  with conn:
    cur = conn.cursor()
    cur.execute(f'SELECT uuid, {str(str_list).replace("[","").replace("]","").replace(stop,", ").replace(stop1,"")} FROM zoning_attributes WHERE {str(str_list).replace("[","").replace("]","").replace(stop," IS NOT NULL AND ").replace(stop1,"")} IS NOT NULL')
    results = cur.fetchall()
    for result in results:
      list_index.append(result[0])
      reg_array.append(list(result[1:]))

  index_list = []
  for id in list_index:
    index_list.append(uuid_list.index(id))

  og_X = scipy.sparse.load_npz('tfidf.npz')
  X = og_X[index_list]
  y = np.asarray(reg_array)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=1)

  model = Sequential([
      Dense(20, input_dim=X.shape[1], activation='relu'),
      Dense(10, activation='relu'), 
      Dense(1, activation='relu'),
      Dense(y.shape[1], activation='relu', use_bias=False, kernel_constraint=NonNeg())
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')

  model.fit(sparse_converter(X_train), y_train, epochs=3, validation_data=(sparse_converter(X_val), y_val), verbose=1)
  model.save(f"full_{param}_model")
  print(model.evaluate(sparse_converter(X_test), y_test))

  model = tf.keras.models.load_model(f"full_{param}_model")
  new_model = Sequential([
      Dense(20, input_dim=og_X.shape[1], activation='relu', weights=model.layers[0].get_weights()),
      Dense(10, activation='relu', weights=model.layers[1].get_weights()), 
      Dense(1, activation='relu', weights=model.layers[2].get_weights()),
  ])
  new_model.compile(optimizer='adam', loss='mean_squared_error')

  conn = create_connection()
  with conn:
    cur = conn.cursor()
    cur.execute(f'ALTER TABLE zoning_attributes ADD COLUMN {param} FLOAT')
    conn.commit()

  def assign_worker(index):
    row = og_X[index]
    imputation = new_model.predict(sparse_converter(row))[0][0]
    conn = create_connection()
    conn.__enter__()
    cur = conn.cursor()
    cur.execute(f"UPDATE zoning_attributes SET {param}={imputation} WHERE uuid='{uuid_list[index]}'")
    conn.commit()
    conn.__exit__()

  tasks = list(range(len(uuid_list)))
  for task in tqdm.tqdm(tasks):
    assign_worker(task)

create_index('alpha') 
create_index('beta') 
create_index('tau') 

S3FS.put('full_alpha_model',f'{S3_PATH}nlp/models/full_alpha_model', recursive=True)
os.remove('full_alpha_model')
S3FS.put('full_beta_model',f'{S3_PATH}nlp/models/full_beta_model', recursive=True)
os.remove('full_beta_model')
S3FS.put('full_tau_model',f'{S3_PATH}nlp/models/full_tau_model', recursive=True)
os.remove('full_tau_model')

S3FS.put('tfidf.npz',f'{S3_PATH}nlp/pickle/tfidf.npz')
os.remove('tfidf.npz')