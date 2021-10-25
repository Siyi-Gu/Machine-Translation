MODEL_PATH=./transferred
DATA_PATH=/mnt/Machine-Translation/processed/hi/hi-en
PREFIX_OUT=hi_test

CUDA_VISIBLE_DEVICES=0 fairseq-generate ${DATA_PATH} \
    --path $MODEL_PATH/checkpoint_best.pt \
    --gen-subset valid \
    -t en_XX -s hi_IN \
    --remove-bpe 'sentencepiece' \
    --batch-size 64 | tee $PREFIX_OUT.out

grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref
