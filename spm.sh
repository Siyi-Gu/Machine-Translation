
SPM="C:/program files/python39/lib/site-packages/sentencepiece"
MODEL=sentence.bpe.model
DATA=C:/Users/nullv/Desktop/2021Autumn/MT/project/processed
TRAIN=train
VALID=valid
SRC=hi_IN
TGT=en_XX
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
