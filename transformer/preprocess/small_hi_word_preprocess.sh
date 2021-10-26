DATA=/mnt/Machine-Translation/transformer/preprocess/small_hi
TRAIN=train
VALID=valid
TEST=test
SRC=hi_IN
TGT=en_XX
NAME=hi-en-word
DEST=/mnt/Machine-Translation/transformer/preprocess/small_hi


fairseq-preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${DATA}/${TRAIN} \
    --validpref ${DATA}/${VALID} \
    --destdir ${DEST}/${NAME} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 70