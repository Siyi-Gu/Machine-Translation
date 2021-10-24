DATA=/mnt/Machine-Translation/processed/hi
TRAIN=train
VALID=valid

SRC=gu_IN
TGT=en_XX
NAME=hi-en
DEST=/mnt/Machine-Translation/processed/hi
DICT=/mnt/mbart.cc25.v2/dict.txt

fairseq-preprocess \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70