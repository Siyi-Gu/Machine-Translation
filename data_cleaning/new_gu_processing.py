from sklearn.model_selection import train_test_split
import string

translation_table = str.maketrans('', '', string.punctuation)

gu_raw = open("data/train.gu", 'r',encoding='utf-8').readlines()
en_raw = open("data/train.en", 'r',encoding='utf-8').readlines()
gu_punct_removed = [line.translate(translation_table) for line in gu_raw ]
en_punct_removed = [line.translate(translation_table) for line in en_raw ]
gu_train, gu_valid = train_test_split(gu_punct_removed,test_size=0.2,random_state=42)
en_train, en_valid = train_test_split(en_punct_removed,test_size=0.2,random_state=42)

file_mapping = {
    'train.gu_IN': gu_train,
    'train.en_XX': en_train,
    'valid.gu_IN': gu_valid,
    'valid.en_XX': en_valid,}

for k, v in file_mapping.items():
    with open(f'./processed/new_gu/{k}', 'w',encoding='utf-8') as fp:
        fp.writelines(v)

