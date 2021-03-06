DATA=/mnt/Machine-Translation/processed/gu
TRAIN=train
VALID=valid
TEST=test
SRC=gu_IN
TGT=en_XX
NAME=gu-en
DEST=/mnt/Machine-Translation/processed/gu
DICT=/mnt/mbart.cc25.v2/dict.txt

fairseq-preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${DATA}/${TRAIN}.spm \
    --validpref ${DATA}/${VALID}.spm \
    --testpref ${DATA}/${TEST}.spm  \
    --destdir ${DEST}/${NAME} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70

