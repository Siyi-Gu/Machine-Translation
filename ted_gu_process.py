import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""https://tmramalho.github.io/science/2020/06/10/fine-tune-neural-translation-models-with-mBART/"""


def pairs(f1,f2):
    data ={}
    data["source"] =[]
    data["target"] =[]
    with open(f1,encoding='utf-8') as src_file, open(f2,encoding='utf-8') as tgt_file:
        for src, tgt in zip(src_file,tgt_file):
            data["source"].append(re.sub("\(.*?\)", "", src.strip())+'\n')
            data["target"].append(re.sub("\(.*?\)", "", tgt.strip())+'\n')
    return data

gu_pairs = pairs("data/TED2020.en-gu.gu","data/TED2020.en-gu.en")

#remove empyty rows
gu_df = pd.DataFrame.from_dict(gu_pairs)
gu_df.replace("", np.nan, inplace=True)
gu_df.dropna(subset=['target'], inplace=True)
gu_df.dropna(subset=['source'], inplace=True)

#train, validation split 
train, valid = train_test_split(gu_df,test_size=0.2,random_state=42)

gu_train = train['source']
en_train = train['target']
gu_valid = valid['source']
en_valid = valid['target']

file_mapping = {
    'train.gu_IN': gu_train,
    'train.en_XX': en_train,
    'valid.gu_IN': gu_valid,
    'valid.en_XX': en_valid,}

for k, v in file_mapping.items():
    with open(f'processed/ted_gu/{k}', 'w',encoding='utf-8') as fp:
        fp.writelines(v)
