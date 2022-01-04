from global_var import *
import pandas as pd 
import numpy as np

###### FIMS – Parent to Children Mapping
# fims = pd.read_sas(read_file("psid/sas/fim12196_gid_BA_2_UBL_wide.sas7bdat") , format='sas7bdat')

def create_code(ids, x):
    if (not np.isnan(x[ids[0]])) or (not np.isnan(x[ids[1]])):
        return int(x[ids[0]]*1000+x[ids[1]])
    else:
        return np.nan

# id_dict = {'CHILD_ID':('ER30001', 'ER30002'), 'A_FATHER_ID':('ER30001_P_AF', 'ER30002_P_AF'), 'A_MOTHER_ID':('ER30001_P_AM', 'ER30002_P_AM'),
#             'FATHER_ID':('ER30001_P_F', 'ER30002_P_F'), 'MOTHER_ID':('ER30001_P_M', 'ER30002_P_M')}

# for id in id_dict:
#     fims[id] = fims.apply(lambda x: create_code(id_dict[id], x), axis=1)

# fims = fims[['CHILD_ID', 'A_FATHER_ID', 'A_MOTHER_ID', 'FATHER_ID', 'MOTHER_ID']]
# write_csv('psid/csv/fims.csv', fims)
fims = pd.read_csv(read_file('psid/csv/fims.csv'))
####### 1990
psid = pd.read_csv(read_file("psid/csv/1990/J301452.csv"))

labels = pd.read_csv(read_file("psid/csv/1990/J301452_labels.txt"))
labels = labels[2:-1]
labels['CODE'] = labels['****** PSID DATA CENTER ************************* '].str.slice(stop=7)
labels['VARIABLE'] = labels['****** PSID DATA CENTER ************************* '].str.slice(start= 10, stop=40).apply(lambda x: x.strip())
labels = labels[['CODE','VARIABLE']].set_index(labels['CODE']).to_dict()['VARIABLE']
print(labels)

psid['CHILD_ID'] = psid.apply(lambda x: create_code(('ER30001', 'ER30002'), x), axis=1)
psid = psid.rename(columns=labels)

child = psid[(psid['AGE OF INDIVIDUAL']<=4) & (psid['YEAR INDIVIDUAL BORN']>=1985)] # Born between 1985-1990
child_ids = child['CHILD_ID']
child_ids = pd.merge(child_ids, fims, how='left', on='CHILD_ID')
print(child_ids)