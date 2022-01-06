from global_var import *
import sqlite3
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\n|d-|____________|_____|\t|---')
def cleanhtml(raw_html):
  try:
    cleantext = ' '.join(re.sub(CLEANR, '', raw_html).split())
    cleantext = re.sub('[^A-Za-z0-9 .,!?:;]', '', cleantext).replace('content is too large for cell.', '')
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

    ## Attributes
    block_title = re.findall('[A-Z][^A-Z]*', filename.split('/')[-1])
    date = re.sub("[^0-9]", "", block_title[-1])
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    state = ''.join(block_title[-10:-8])
    muni = ' '.join(block_title[:-10])

    ## Text Data
    df = pd.read_excel(f's3://{filename}')
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(columns = ['NodeId','Url'])
    df = df.applymap(lambda x: cleanhtml(x)).fillna('')
    
    df = df[~(df['Title'].str.contains('table'))].reset_index(drop=True)
  
    def text_creator(df):
      df['raw_text'] = df.apply(lambda x: f'{x["Title"]} {x["Subtitle"]} {x["Content"]} ', axis=1)
      text = df['raw_text'].str.cat()
      return text
    
    raw_text = text_creator(df)

    zoning_text = None

    try:
      app_idx = df[df['Title'].str.contains('appendix')].iloc[0].name
      body = df.iloc[:app_idx]
      appendix = df.iloc[app_idx:]

      appendices = appendix[appendix['Title'].str.contains('appendix')]
      print(appendices)

    except:
      body = df
      appendix = None
      
    # print(body)

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