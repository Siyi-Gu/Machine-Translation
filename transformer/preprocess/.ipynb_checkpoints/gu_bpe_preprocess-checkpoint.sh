DATA=/mnt/Machine-Translation/transformer/preprocess/gu
TRAIN=train
VALID=valid
TEST=test
SRC=gu_IN
TGT=en_XX
NAME=gu-en-bpe
DEST=/mnt/Machine-Translation/transformer/preprocess/gu
DICT=/mnt/mbart.cc25.v2/dict.txt

fairseq-preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${DATA}/${TRAIN}.spm \
    --validpref ${DATA}/${VALID}.spm \
    --testpref ${DATA}/${TEST}.spm \
    --destdir ${DEST}/${NAME} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70
    
# fairseq-preprocess --source-lang fr --target-lang en \
#     --trainpref $DATA_ROOT/fren.train --validpref $DATA_ROOT/fren.dev --testpref $DATA_ROOT/fren.test \
#     --destdir $DATA_PATH