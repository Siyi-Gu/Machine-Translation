import re
import pandas as pd
from sklearn.model_selection import train_test_split

def pairs():
    lines = open('wikititles-v1.gu-en.tsv', encoding='utf-8').read().strip().split('\n')
    #get rid of non-translation
    lines = [l for l in lines if not re.match("[A-Za-z]",l[0])]
    # Split every line into pairs and normalize
    pairs = [[s+'\n' for s in l.split('\t')] for l in lines]
    return pairs

gu_df=pd.DataFrame(pairs())
gu_df.columns = ['source','target']

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
    with open(f'processed/gu/{k}', 'w',encoding='utf-8') as fp:
        fp.writelines(v)
