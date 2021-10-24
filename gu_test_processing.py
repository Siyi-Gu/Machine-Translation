import re
import pandas as pd
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

gu_test = gu_df['source']
en_test = gu_df['target']

file_mapping = {
    'test.gu_IN': gu_test,
    'test.en_XX': en_test}

for k, v in file_mapping.items():
    with open(f'processed/gu/{k}', 'w',encoding='utf-8') as fp:
        fp.writelines(v)
