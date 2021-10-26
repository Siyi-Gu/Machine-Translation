import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string

translation_table = str.maketrans('', '', string.punctuation)
def pairs():
    lines = open('data/wikititles-v1.gu-en.tsv', encoding='utf-8').read().strip().split('\n')
    #get rid of non-translation
    lines = [l.lower().translate(translation_table) for l in lines if not re.match("[A-Za-z]",l[0])]
    # Split every line into pairs and normalize
    pairs = [[s+'\n' for s in l.split('\t')] for l in lines]
    return pairs

gu_df=pd.DataFrame(pairs())
gu_df.columns = ['source','target']
gu_df.replace("", np.nan, inplace=True)
gu_df.replace("\n", np.nan, inplace=True)
gu_df.dropna(subset=['target'], inplace=True)
gu_df.dropna(subset=['source'], inplace=True)
gu_df = gu_df.drop_duplicates()

#train, validation, test split 
train, sub = train_test_split(gu_df,test_size=0.2,random_state=42)
valid, test = train_test_split(sub,test_size=0.5,random_state=42)

gu_train = train['source']
en_train = train['target']
gu_valid = valid['source']
en_valid = valid['target']
gu_test = test['source']
en_test = test['target']

file_mapping = {
    'train.gu_IN': gu_train,
    'train.en_XX': en_train,
    'valid.gu_IN': gu_valid,
    'valid.en_XX': en_valid,
    'test.gu_IN': gu_test,
    'test.en_XX': en_test}

for k, v in file_mapping.items():
    with open(f'processed/gu/{k}', 'w',encoding='utf-8') as fp:
        fp.writelines(v)
