from global_var import *
import sqlite3
from sqlite3 import Error
from contextlib import closing
import math
import csv
import tqdm
import sys
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, Array
import re
import random
from sentence_transformers import SentenceTransformer, losses, InputExample, util, evaluation
from torch.utils.data import DataLoader

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_row(conn, row):
    cur = conn.cursor()
    cur.execute(''' CREATE TABLE IF NOT EXISTS zoning (
                      place TEXT NOT NULL,
                      state TEXT NOT NULL,
                      year INTEGER NOT NULL, 
                      month INTEGER NOT NULL, 
                      day INTEGER NOT NULL, 
                      zone_text TEXT NOT NULL 
                    ); ''')
    conn.commit()
    cur.execute("INSERT INTO zoning (place, state, year, month, day, zone_text) values (?,?,?,?,?,?);", row)
    conn.commit()

database = 'zoning_corpus.db'

conn = create_connection(database)
with conn:
  cur = conn.cursor()
  done_list = set(cur.execute('SELECT place, state, year, month, day FROM zoning').fetchall())

zoning_phrases = ['development', 'development code', 'zoning', 'architectural', 'infill', 'density', 'housing', 'land', 'growth areas', 'building', 'preservation', 'historic', 'land use', 'neighborhood', 'subdivision', 'planning', 'property maintenace']
directory = S3FS.ls(f'{S3_PATH}nlp/csv/municode')
tasks = directory[:int(.5*len(directory))] #random.sample(directory, k=3)

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

    conn = create_connection(database)
    with conn:
      row = [muni, state, year, month, day, zone_text]
      create_row(conn, row)

if __name__ == "__main__":
  pool= Pool()
  for _ in tqdm.tqdm(pool.imap_unordered(loop, tasks), total=len(tasks)):
    pass