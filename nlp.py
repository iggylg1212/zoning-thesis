from global_var import *
from contextlib import closing
from config import * 
import math
import csv
import tqdm
import pymysql
import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, Array
import re
import random
from sentence_transformers import SentenceTransformer, losses, InputExample, util, evaluation
from torch.utils.data import DataLoader

def create_connection():
    conn = None
    try:
        conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
    except:
        print('Error')

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
                      urbanized BOOL DEFAULT 0
                    ); ''')
    conn.commit()
    cur.execute("INSERT INTO zoning (place, state, year, month, day, zone_text) values (%s,%s,%s,%s,%s,%s);", row)
    conn.commit()

conn = create_connection()
with conn:
  cur = conn.cursor()
  cur.execute('SELECT place, state, year, month, day FROM zoning')
  done_list = set(cur.fetchall())

zoning_phrases = ['development', 'development code', 'zoning', 'architectural', 'infill', 'density', 'housing', 'land', 'growth areas', 'building', 'preservation', 'historic', 'land use', 'neighborhood', 'subdivision', 'planning', 'property maintenace']
directory = S3FS.ls(f'{S3_PATH}nlp/csv/municode')
tasks = directory #random.sample(directory, k=3)

model_name = 'paraphrase-MiniLM-L3-v2'
model_save_path = f'continue_training-{model_name}'
# train_batch_size = 16
# num_epochs = 4

# def create_train(df):
#   train_samples = []
#   dev_samples = []
#   for idx, row in df.iterrows():
#     for column in df.columns.to_list()[1:]:
#       inp_example = InputExample(texts=[row['Text'], column], label=float(row[column]))
#       if random.randrange(0, 10) < 2:
#         dev_samples.append(inp_example)
#       else:
#         train_samples.append(inp_example)
#   return train_samples, dev_samples

# fine_tuning = pd.read_csv(f'{S3_PATH}nlp/csv/fine_tuning_semantic.csv')
# train_samples, dev_samples = create_train(fine_tuning)

# model = SentenceTransformer(model_name)
# train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.CosineSimilarityLoss(model=model)
# evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
# warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
# model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=num_epochs, evaluation_steps=1000, warmup_steps=warmup_steps, output_path=model_save_path)

model = SentenceTransformer(model_save_path)
# test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)

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

# # Adding GISJOINs to zoning_corpus
# us_state_to_abbrev = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC", "American Samoa": "AS", "Guam": "GU", "Northern Mariana Islands": "MP", "Puerto Rico": "PR", "United States Minor Outlying Islands": "UM", "U.S. Virgin Islands": "VI"}
#
# county_code= pd.read_csv(f'{S3_PATH}nlp/csv/2010_population/nhgis0017_ds120_1990_county.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','ANPSADPI']]
# place_code = pd.read_csv(f'{S3_PATH}nlp/csv/2010_population/nhgis0017_ds120_1990_place.csv', encoding = "ISO-8859-1")[['GISJOIN','STATE','PLACE']]
# codes = pd.concat([place_code, county_code], axis=0)
# codes['STATE'] = codes['STATE'].replace(us_state_to_abbrev)
# codes['PLACE'] = codes['PLACE'].fillna('').apply(lambda x: ' '.join(x.split(' ')[:-1]))
# codes['ANPSADPI'] = codes['ANPSADPI'].fillna('')
#
# place_list = str(set(codes[codes['PLACE']!='']['GISJOIN'])).replace('}',')').replace('{','(')
# county_list = str(set(codes[codes['ANPSADPI']!='']['GISJOIN'])).replace('}',')').replace('{','(')
#
# conn = create_connection()
# with conn:
#   cur = conn.cursor()
  # for id, row in codes[codes['ANPSADPI']!=''].iterrows():
  #   county = row['ANPSADPI'].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin = '{row['GISJOIN']}' WHERE place='{county}' AND state='{row['STATE']}'")
  #   conn.commit()
  # for id, row in codes[codes['PLACE']!=''].iterrows():
  #   place = row["PLACE"].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin = '{row['GISJOIN']}' WHERE place ='{place}' AND state='{row['STATE']}'")
  #   conn.commit()
  # len = len(codes[codes['PLACE']!=''])
  # number = 0
  # for id, row in codes[codes['PLACE']!=''].iterrows():
  #   number+=1
  #   print(round(number/len, 4))
  #   place = row["PLACE"].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin='{row['GISJOIN']}' WHERE gisjoin is NULL and place LIKE '%{place}%' AND state='{row['STATE']}'")
  #   conn.commit()
  # for id, row in codes[(codes['ANPSADPI']!='') & (codes['STATE']=='LA')].iterrows(): # Correcting 'police jury'
  #   county = row['ANPSADPI'].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin='{row['GISJOIN']}' WHERE gisjoin is NULL and place LIKE '%{county}%' AND state='LA'")
  #   conn.commit()

  # cur.execute('SELECT place, state FROM zoning WHERE gisjoin IS NULL')
  # null_set = set(cur.fetchall())
  # dataframe = pd.DataFrame(columns=['og_place','og_state'])
  # for set in null_set:
  #   dataframe = dataframe.append({'og_place':set[0],'og_state':set[1]}, ignore_index=True)
  # dataframe.to_csv('broken.csv')

  # corrected = pd.read_csv('broken.csv').fillna('')
  # for id, row in corrected.iterrows():
  #   og_place = row['og_place'].replace("'","''")
  #   new_place = row['new_place'].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET place='{new_place}', state='{row['new_state']}' WHERE place='{og_place}' AND state='{row['og_state']}'")
  #   conn.commit()

  # len = len(codes[codes['ANPSADPI']!=''])+len(codes[codes['PLACE']!=''])
  # number = 0 
  # for id, row in codes[codes['ANPSADPI']!=''].iterrows():
  #   number += 1
  #   print(round(number/len,4))
  #   county = row['ANPSADPI'].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin = '{row['GISJOIN']}' WHERE gisjoin is NULL AND place='{county}' AND state='{row['STATE']}'")
  #   conn.commit()
  # for id, row in codes[codes['PLACE']!=''].iterrows():
  #   number += 1
  #   print(round(number/len,4))
  #   place = row["PLACE"].replace("'","''")
  #   cur.execute(f"UPDATE zoning SET gisjoin = '{row['GISJOIN']}' WHERE gisjoin is NULL AND place ='{place}' AND state='{row['STATE']}'")
  #   conn.commit()

  # cur.execute(f"UPDATE zoning SET govtype = 'county' WHERE gisjoin in {county_list}")
  # cur.execute(f"UPDATE zoning SET govtype = 'place' WHERE gisjoin in {place_list}")
  # conn.commit()

# # Finding places and counties in urbanized areas
# def import_shpfile(path):
#   shp_files = ['.dbf','.prj','.sbn','.sbx','.shp','.shp.xml','.shx']
#   for file in shp_files:
#     S3FS.get(f'{S3_PATH}{path}{file}', f'temporary{file}')
#   df = gpd.read_file('temporary.shp', crs='EPSG:4326')
#   for file in shp_files:
#     os.remove(f'temporary{file}')
#   return df
#
# urbs = import_shpfile('nlp/gis/urb_area_1990/US_urb_area_1990').to_crs('EPSG:4326')
#
# county = import_shpfile('nlp/gis/county_1990/US_county_1990').to_crs('EPSG:4326')
# county['source'] = 'county'
# places = import_shpfile('nlp/gis/us_place_1990/US_place_1990').to_crs('EPSG:4326')
# places['source'] = 'place'
# govs = pd.concat([county, places])[['GISJOIN','source','geometry']]
#
# urb_gov = gpd.GeoDataFrame(columns=['GISJOIN','UACODE','source','geometry'], crs='EPSG:4326')
# for row in list(range(len(urbs))):
#     print(round(row/len(urbs),4))
#     city = urbs.iloc[row][['UACODE','geometry']]
#     govs['bool'] = govs.intersects(city['geometry'])
#     govs['UACODE'] = city['UACODE']
#     set = govs[govs['bool'] == True]
#     urb_gov = pd.concat([urb_gov, set])
#
# urb_gov.to_file('urb_gov.gpkg')
# S3FS.put('urb_gov.gpkg',f'{S3_PATH}nlp/gis/urb_gov_1990.gpkg')
# os.remove('urb_gov.gpkg')

# # Find urban entries
# S3FS.get(f'{S3_PATH}nlp/gis/urb_gov_1990.gpkg', 'urb_gov.gpkg')
# urb_gov = gpd.read_file('urb_gov.gpkg').to_crs('EPSG:4326')
# os.remove('urb_gov.gpkg')
# urb_joins = str(set(urb_gov['GISJOIN'])).replace('{','').replace('}','')
#
# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute(f"UPDATE zoning SET urbanized=1 WHERE gisjoin IN ({urb_joins})")
#     conn.commit()

