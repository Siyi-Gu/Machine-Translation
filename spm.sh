
SPM=/Users/sig/sentencepiece/build/src/spm_encode
MODEL=/Users/sig/Documents/Sy/mbart.cc25.v2/sentence.bpe.model
DATA=/Users/sig/Documents/Sy/Machine-Translation/processed/gu
TRAIN=train
VALID=valid
SRC=gu_IN
TGT=en_XX
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
