from global_var import *
import sqlite3
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\n|____________|_____|\t|---')
def cleanhtml(raw_html):
  try:
    cleantext = ' '.join(re.sub(CLEANR, '', raw_html).split())
  except:
    cleantext = raw_html  
  return cleantext

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def semantic_similarity(embed, sen):
    sen = model.encode(sen)
    cosine_scores = util.pytorch_cos_sim(embed, sen)
    return float(cosine_scores[0][0])

tot_len = len(S3FS.ls(f'{S3_PATH}/nlp/csv/municode'))
ticker = 0
for filename in S3FS.ls(f'{S3_PATH}/nlp/csv/municode'):
    ticker += 1
    # print(round(ticker/tot_len , 3), end='\r', flush=True)
  
    block_title = re.findall('([A-Z])', filename.split('/')[-1])
    print(block_title)


    df = pd.read_excel(f's3://{filename}')
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(columns = ['NodeId','Url'])
    df['Content'] = df['Content'].apply(lambda x: cleanhtml(x))
    df = df[(df['Content']!= 'Content is too large for cell.') & (df['Content'].notna())].reset_index(drop=True)
    
    # print(df)


# for table in conn.execute('''select name  from sqlite_master where type='table' '''):
#     table = table[0]
#     query_list = []
#     ticker = 0 
#     rows = conn.execute(f'''ALTER TABLE {table} 
#     RENAME COLUMN 'index' to Url
#     RENAME COLUMN 'INTEGER' to Url

#     ''')
#     for row in conn.execute(f'''select * from {table}'''):
#         # query_list.append( (row[3], row[4]) ) 
#         print(row)
#         break
#     break